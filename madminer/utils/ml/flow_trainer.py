from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np

import torch
from torch import tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils import clip_grad_norm_

from madminer.utils.ml.utils import check_for_nans_in_parameters


class SmallGoldDataset(torch.utils.data.Dataset):
    """ """

    def __init__(self, theta0=None, x=None, t_xz0=None):
        self.n = theta0.shape[0]

        placeholder = torch.stack([tensor([0.0]) for _ in range(self.n)])

        assert x is not None
        assert theta0 is not None

        self.theta0 = theta0
        self.x = x
        self.t_xz0 = placeholder if t_xz0 is None else t_xz0

        assert len(self.theta0) == self.n
        assert len(self.x) == self.n
        assert len(self.t_xz0) == self.n

    def __getitem__(self, index):
        return (self.theta0[index], self.x[index], self.t_xz0[index])

    def __len__(self):
        return self.n


def train_flow_model(
    model,
    loss_functions,
    theta0s=None,
    xs=None,
    t_xz0s=None,
    loss_weights=None,
    loss_labels=None,
    batch_size=64,
    trainer="adam",
    initial_learning_rate=0.001,
    final_learning_rate=0.0001,
    nesterov_momentum=None,
    n_epochs=50,
    clip_gradient=100.0,
    run_on_gpu=True,
    double_precision=False,
    validation_split=0.2,
    early_stopping=True,
    early_stopping_patience=None,
    grad_x_regularization=None,
    learning_curve_folder=None,
    learning_curve_filename=None,
    verbose="some",
):
    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.double if double_precision else torch.float

    # Move model to device
    model = model.to(device, dtype)

    # Convert to Tensor
    if theta0s is not None:
        theta0s = torch.stack([tensor(i, requires_grad=True) for i in theta0s])
    if xs is not None:
        xs = torch.stack([tensor(i) for i in xs])
    if t_xz0s is not None:
        t_xz0s = torch.stack([tensor(i) for i in t_xz0s])

    # Dataset
    dataset = SmallGoldDataset(theta0s, xs, t_xz0s)

    # Train / validation split
    if validation_split is not None:
        assert 0.0 < validation_split < 1.0, "Wrong validation split: {}".format(validation_split)

        n_samples = len(dataset)
        indices = list(range(n_samples))
        split = int(np.floor(validation_split * n_samples))
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        validation_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, pin_memory=run_on_gpu)
        validation_loader = DataLoader(
            dataset, sampler=validation_sampler, batch_size=batch_size, pin_memory=run_on_gpu
        )
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=run_on_gpu)

    # Optimizer
    if trainer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    elif trainer == "amsgrad":
        optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, amsgrad=True)
    elif trainer == "sgd":
        if nesterov_momentum is None:
            optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate)
        else:
            optimizer = optim.SGD(
                model.parameters(), lr=initial_learning_rate, nesterov=True, momentum=nesterov_momentum
            )
    else:
        raise ValueError("Unknown trainer {}".format(trainer))

    # Early stopping
    early_stopping = early_stopping and (validation_split is not None) and (n_epochs > 1)
    early_stopping_best_val_loss = None
    early_stopping_best_model = None
    early_stopping_epoch = None

    # Loss functions
    n_losses = len(loss_functions)

    if loss_weights is None:
        loss_weights = [1.0] * n_losses

    # Regularization
    if grad_x_regularization is not None:
        raise NotImplementedError("Flow training does not support grad_x regularization yet!")

    # Losses over training
    individual_losses_train = []
    individual_losses_val = []
    total_losses_train = []
    total_losses_val = []
    total_val_loss = None

    # Verbosity
    n_epochs_verbose = None
    if verbose == "all":  # Print output after every epoch
        n_epochs_verbose = 1
    elif verbose == "some":  # Print output after 10%, 20%, ..., 100% progress
        n_epochs_verbose = max(int(round(n_epochs / 10, 0)), 1)

    # Loop over epochs
    for epoch in range(n_epochs):

        # Training
        model.train()
        individual_train_loss = np.zeros(n_losses)
        total_train_loss = 0.0

        # Learning rate decay
        if n_epochs > 1:
            lr = initial_learning_rate * (final_learning_rate / initial_learning_rate) ** float(
                epoch / (n_epochs - 1.0)
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # Loop over batches
        for i_batch, (theta0, x, t_xz0) in enumerate(train_loader):
            theta0 = theta0.to(device, dtype)
            x = x.to(device, dtype)
            try:
                t_xz0 = t_xz0.to(device, dtype)
            except NameError:
                pass

            optimizer.zero_grad()

            # Forward pass
            if t_xz0 is not None:
                _, log_likelihood, score = model.log_likelihood_and_score(theta0, x)
            else:
                _, log_likelihood = model.log_likelihood(theta0, x)
                score = None

            # Evaluate loss
            losses = [fn(log_likelihood, score, t_xz0) for fn in loss_functions]
            loss = loss_weights[0] * losses[0]
            for _w, _l in zip(loss_weights[1:], losses[1:]):
                loss += _w * _l

            for i, individual_loss in enumerate(losses):
                individual_train_loss[i] += individual_loss.item()
            total_train_loss += loss.item()

            # Calculate gradient and update optimizer
            loss.backward()

            # Clip gradients
            if clip_gradient is not None:
                clip_grad_norm_(model.parameters(), clip_gradient)

            # Check for NaNs
            if check_for_nans_in_parameters(model):
                logging.warning("NaNs in parameters or gradients, stopping training!")
                break

            # Optimizer step
            optimizer.step()

        individual_train_loss /= len(train_loader)
        total_train_loss /= len(train_loader)

        total_losses_train.append(total_train_loss)
        individual_losses_train.append(individual_train_loss)

        # Validation
        if validation_split is None:
            if n_epochs_verbose is not None and n_epochs_verbose > 0 and (epoch + 1) % n_epochs_verbose == 0:
                logging.info(
                    "  Epoch %d: train loss %.2f (%s)"
                    % (epoch + 1, total_losses_train[-1], individual_losses_train[-1])
                )
            continue

        # with torch.no_grad():
        model.eval()
        individual_val_loss = np.zeros(n_losses)
        total_val_loss = 0.0

        for i_batch, (theta0, x, t_xz0) in enumerate(validation_loader):
            theta0 = theta0.to(device, dtype)
            x = x.to(device, dtype)
            try:
                t_xz0 = t_xz0.to(device, dtype)
            except NameError:
                pass

            # Forward pass
            if t_xz0 is not None:
                _, log_likelihood, score = model.log_likelihood_and_score(theta0, x)
            else:
                _, log_likelihood = model.log_likelihood(theta0, x)
                score = None

            # Evaluate loss
            losses = [fn(log_likelihood, score, t_xz0) for fn in loss_functions]
            loss = loss_weights[0] * losses[0]
            for _w, _l in zip(loss_weights[1:], losses[1:]):
                loss += _w * _l

            for i, individual_loss in enumerate(losses):
                individual_val_loss[i] += individual_loss.item()
            total_val_loss += loss.item()

        individual_val_loss /= len(validation_loader)
        total_val_loss /= len(validation_loader)

        total_losses_val.append(total_val_loss)
        individual_losses_val.append(individual_val_loss)

        # Early stopping: best epoch so far?
        if early_stopping:
            if early_stopping_best_val_loss is None or total_val_loss < early_stopping_best_val_loss:
                early_stopping_best_val_loss = total_val_loss
                early_stopping_best_model = model.state_dict()
                early_stopping_epoch = epoch

        # Print out information
        if n_epochs_verbose is not None and n_epochs_verbose > 0 and (epoch + 1) % n_epochs_verbose == 0:
            if early_stopping and epoch == early_stopping_epoch:
                logging.info(
                    "  Epoch %d: train loss %.2f (%s), validation loss %.2f (%s) (*)"
                    % (
                        epoch + 1,
                        total_losses_train[-1],
                        individual_losses_train[-1],
                        total_losses_val[-1],
                        individual_losses_val[-1],
                    )
                )
            else:
                logging.info(
                    "  Epoch %d: train loss %.2f (%s), validation loss %.2f (%s)"
                    % (
                        epoch + 1,
                        total_losses_train[-1],
                        individual_losses_train[-1],
                        total_losses_val[-1],
                        individual_losses_val[-1],
                    )
                )

        # Early stopping: actually stop training
        if early_stopping and early_stopping_patience is not None:
            if epoch - early_stopping_epoch >= early_stopping_patience > 0:
                logging.info("No improvement for %s epochs, stopping training", epoch - early_stopping_epoch)
                break

    # Early stopping: back to best state
    if early_stopping:
        if early_stopping_best_val_loss < total_val_loss:
            logging.info(
                "Early stopping after epoch %s, with loss %.2f compared to final loss %.2f",
                early_stopping_epoch + 1,
                early_stopping_best_val_loss,
                total_val_loss,
            )
            model.load_state_dict(early_stopping_best_model)
        else:
            logging.info("Early stopping did not improve performance")

    # Save learning curve
    if learning_curve_folder is not None and learning_curve_filename is not None:

        np.save(learning_curve_folder + "/loss_train" + learning_curve_filename + ".npy", total_losses_train)
        if validation_split is not None:
            np.save(learning_curve_folder + "/loss_val" + learning_curve_filename + ".npy", total_losses_val)

        if loss_labels is not None:
            individual_losses_train = np.array(individual_losses_train)
            individual_losses_val = np.array(individual_losses_val)

            for i, label in enumerate(loss_labels):
                np.save(
                    learning_curve_folder + "/loss_" + label + "_train" + learning_curve_filename + ".npy",
                    individual_losses_train[:, i],
                )
                if validation_split is not None:
                    np.save(
                        learning_curve_folder + "/loss_" + label + "_val" + learning_curve_filename + ".npy",
                        individual_losses_val[:, i],
                    )

    logging.info("Finished training")

    return total_losses_train, total_losses_val


def evaluate_flow_model(model, theta0s=None, xs=None, evaluate_score=False, run_on_gpu=True, double_precision=False):
    """

    Parameters
    ----------
    model :
        
    theta0s :
         (Default value = None)
    xs :
         (Default value = None)
    evaluate_score :
         (Default value = False)
    run_on_gpu :
         (Default value = True)
    double_precision :
         (Default value = False)

    Returns
    -------

    """
    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.double if double_precision else torch.float

    # Balance theta0 and theta1
    n_thetas = len(theta0s)

    # Prepare data
    n_xs = len(xs)
    theta0s = torch.stack([tensor(theta0s[i % n_thetas], requires_grad=True) for i in range(n_xs)])
    xs = torch.stack([tensor(i) for i in xs])

    model = model.to(device, dtype)
    theta0s = theta0s.to(device, dtype)
    xs = xs.to(device, dtype)

    # Evaluate estimator with score:
    if evaluate_score:
        model.eval()

        _, log_p_hat, t_hat = model.log_likelihood_and_score(theta0s, xs)

        log_p_hat = log_p_hat.detach().numpy().flatten()
        t_hat = t_hat.detach().numpy().flatten()

    # Evaluate estimator without score:
    else:
        with torch.no_grad():
            model.eval()

            _, log_p_hat = model.log_likelihood(theta0s, xs)

            log_p_hat = log_p_hat.detach().numpy().flatten()
            t_hat = None

    return log_p_hat, t_hat
