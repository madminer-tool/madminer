from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np

import torch
from torch import tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils import clip_grad_norm_


class LocalScoreDataset(torch.utils.data.Dataset):
    """ """

    def __init__(self, x, t_xz):
        self.n = x.shape[0]

        self.x = x
        self.t_xz = t_xz

        assert len(self.t_xz) == self.n

    def __getitem__(self, index):
        return (self.x[index], self.t_xz[index])

    def __len__(self):
        return self.n


def train_local_score_model(
    model,
    loss_functions,
    xs,
    t_xzs,
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
    xs = torch.stack([tensor(i) for i in xs])
    t_xzs = torch.stack([tensor(i) for i in t_xzs])

    # Dataset
    dataset = LocalScoreDataset(xs, t_xzs)

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
        n_losses += 1
        loss_weights.append(grad_x_regularization)
        loss_labels.append("l2_grad_x")

    # Losses over training
    individual_losses_train = []
    individual_losses_val = []
    total_losses_train = []
    total_losses_val = []

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
        for i_batch, (x, t_xz) in enumerate(train_loader):
            x = x.to(device, dtype)
            t_xz = t_xz.to(device, dtype)

            optimizer.zero_grad()

            # Forward pass
            if grad_x_regularization is None:
                t_hat = model(x)
                x_gradient = None
            else:
                t_hat, x_gradient = model(x, return_grad_x=True)

            # Evaluate loss
            losses = [loss_function(t_hat, t_xz) for loss_function in loss_functions]
            if grad_x_regularization is not None:
                losses.append(torch.mean(torch.sum(x_gradient ** 2, dim=1)))

            loss = loss_weights[0] * losses[0]
            for _w, _l in zip(loss_weights[1:], losses[1:]):
                loss += _w * _l

            for i, individual_loss in enumerate(losses):
                individual_train_loss[i] += individual_loss.item()
            total_train_loss += loss.item()

            # Calculate gradient and update optimizer
            loss.backward()
            optimizer.step()

            # Clip gradients
            if clip_gradient is not None:
                clip_grad_norm_(model.parameters(), clip_gradient)

        individual_train_loss /= len(train_loader)
        total_train_loss /= len(train_loader)

        total_losses_train.append(total_train_loss)
        individual_losses_train.append(individual_train_loss)

        # Validation
        if validation_split is None:
            if n_epochs_verbose is not None and n_epochs_verbose > 0 and (epoch + 1) % n_epochs_verbose == 0:
                individual_loss_string = ""
                for i, (label, value) in enumerate(zip(loss_labels, individual_losses_train[-1])):
                    if i > 0:
                        individual_loss_string += ", "
                    individual_loss_string += "{}: {:.4f}".format(label, value)

                logging.info(
                    "  Epoch %d: train loss %.4f (%s)" % (epoch + 1, total_losses_train[-1], individual_loss_string)
                )
            continue

        with torch.no_grad():
            model.eval()
            individual_val_loss = np.zeros(n_losses)
            total_val_loss = 0.0

            for i_batch, (x, t_xz) in enumerate(validation_loader):
                x = x.to(device, dtype)
                t_xz = t_xz.to(device, dtype)

                # Evaluate loss
                t_hat = model(x)

                losses = [loss_function(t_hat, t_xz) for loss_function in loss_functions]
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
            individual_loss_string_train = ""
            individual_loss_string_val = ""
            for i, (label, value_train, value_val) in enumerate(
                zip(loss_labels, individual_losses_train[-1], individual_losses_val[-1])
            ):
                if i > 0:
                    individual_loss_string_train += ", "
                    individual_loss_string_val += ", "
                individual_loss_string_train += "{}: {:.4f}".format(label, value_train)
                individual_loss_string_val += "{}: {:.4f}".format(label, value_val)

            if early_stopping and epoch == early_stopping_epoch:
                logging.info(
                    "  Epoch %d: train loss %.4f (%s)", epoch + 1, total_losses_train[-1], individual_loss_string_train
                )
                logging.info("            val. loss  %.4f (%s) (*)", total_losses_val[-1], individual_loss_string_val)
            else:
                logging.info(
                    "  Epoch %d: train loss %.4f (%s)", epoch + 1, total_losses_train[-1], individual_loss_string_train
                )
                logging.info("            val. loss  %.4f (%s)", total_losses_val[-1], individual_loss_string_val)

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


def evaluate_local_score_model(model, xs=None, run_on_gpu=True, double_precision=False, return_grad_x=False):
    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.double if double_precision else torch.float

    # Prepare data
    xs = torch.stack([tensor(i) for i in xs])

    model = model.to(device, dtype)
    xs = xs.to(device, dtype)

    # Evaluate networks
    if return_grad_x:
        model.eval()
        t_hat, x_gradients = model(xs, return_grad_x=True)
    else:
        with torch.no_grad():
            model.eval()
            t_hat = model(xs)
        x_gradients = None

    # Get data and return
    t_hat = t_hat.detach().numpy()

    if return_grad_x:
        x_gradients = x_gradients.detach().numpy()
        return t_hat, x_gradients

    return t_hat
