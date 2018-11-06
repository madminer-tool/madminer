from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np

import torch
from torch import tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils import clip_grad_norm_

from madminer.utils.ml.models.ratio import ParameterizedRatioEstimator, DoublyParameterizedRatioEstimator


class GoldDataset(torch.utils.data.Dataset):
    """ """

    def __init__(self, theta0=None, theta1=None, x=None, y=None, r_xz=None, t_xz0=None, t_xz1=None):
        self.n = theta0.shape[0]

        placeholder = torch.stack([tensor([0.0]) for _ in range(self.n)])

        assert x is not None
        assert theta0 is not None

        self.theta0 = theta0
        self.theta1 = placeholder if theta1 is None else theta1
        self.x = x
        self.y = placeholder if y is None else y
        self.r_xz = placeholder if r_xz is None else r_xz
        self.t_xz0 = placeholder if t_xz0 is None else t_xz0
        self.t_xz1 = placeholder if t_xz1 is None else t_xz1

        assert len(self.theta0) == self.n
        assert len(self.theta1) == self.n
        assert len(self.x) == self.n
        assert len(self.y) == self.n
        assert len(self.r_xz) == self.n
        assert len(self.t_xz0) == self.n
        assert len(self.t_xz1) == self.n

    def __getitem__(self, index):
        return (
            self.theta0[index],
            self.theta1[index],
            self.x[index],
            self.y[index],
            self.r_xz[index],
            self.t_xz0[index],
            self.t_xz1[index],
        )

    def __len__(self):
        return self.n


def train_ratio_model(
    model,
    loss_functions,
    method_type="parameterized",
    theta0s=None,
    theta1s=None,
    xs=None,
    ys=None,
    r_xzs=None,
    t_xz0s=None,
    t_xz1s=None,
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
    if theta1s is not None:
        theta1s = torch.stack([tensor(i, requires_grad=True) for i in theta1s])
    if xs is not None:
        xs = torch.stack([tensor(i) for i in xs])
    if ys is not None:
        ys = torch.stack([tensor(i) for i in ys])
    if r_xzs is not None:
        r_xzs = torch.stack([tensor(i) for i in r_xzs])
    if t_xz0s is not None:
        t_xz0s = torch.stack([tensor(i) for i in t_xz0s])
    if t_xz1s is not None:
        t_xz1s = torch.stack([tensor(i) for i in t_xz1s])

    # Dataset
    dataset = GoldDataset(theta0s, theta1s, xs, ys, r_xzs, t_xz0s, t_xz1s)

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
        for i_batch, (theta0, theta1, x, y, r_xz, t_xz0, t_xz1) in enumerate(train_loader):
            theta0 = theta0.to(device, dtype)
            x = x.to(device, dtype)
            y = y.to(device, dtype)
            try:
                theta1 = theta1.to(device, dtype)
            except NameError:
                pass
            try:
                r_xz = r_xz.to(device, dtype)
            except NameError:
                pass
            try:
                t_xz0 = t_xz0.to(device, dtype)
            except NameError:
                pass
            try:
                t_xz1 = t_xz1.to(device, dtype)
            except NameError:
                pass

            optimizer.zero_grad()

            # Forward pass
            if grad_x_regularization is None:
                x_gradient = None
                if method_type == "parameterized":
                    s_hat, log_r_hat, t_hat0 = model(theta0, x)
                    t_hat1 = None
                elif method_type == "doubly_parameterized":
                    s_hat, log_r_hat, t_hat0, t_hat1 = model(theta0, theta1, x)
                else:
                    raise ValueError("Unknown method type {}".format(method_type))
            else:
                if method_type == "parameterized":
                    s_hat, log_r_hat, t_hat0, x_gradient = model(theta0, x, return_grad_x=True)
                    t_hat1 = None
                elif method_type == "doubly_parameterized":
                    s_hat, log_r_hat, t_hat0, t_hat1, x_gradient = model(theta0, theta1, x, return_grad_x=True)
                else:
                    raise ValueError("Unknown method type {}".format(method_type))

            # Evaluate loss
            losses = [
                loss_function(s_hat, log_r_hat, t_hat0, t_hat1, y, r_xz, t_xz0, t_xz1)
                for loss_function in loss_functions
            ]
            if grad_x_regularization is not None:
                losses.append(torch.mean(x_gradient ** 2))

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
                logging.info(
                    "  Epoch %d: train loss %.2f (%s)"
                    % (epoch + 1, total_losses_train[-1], individual_losses_train[-1])
                )
            continue

        # with torch.no_grad():
        model.eval()
        individual_val_loss = np.zeros(n_losses)
        total_val_loss = 0.0

        for i_batch, (theta0, theta1, x, y, r_xz, t_xz0, t_xz1) in enumerate(validation_loader):
            theta0 = theta0.to(device, dtype)
            x = x.to(device, dtype)
            y = y.to(device, dtype)
            try:
                theta1 = theta1.to(device, dtype)
            except NameError:
                pass
            try:
                r_xz = r_xz.to(device, dtype)
            except NameError:
                pass
            try:
                t_xz0 = t_xz0.to(device, dtype)
            except NameError:
                pass
            try:
                t_xz1 = t_xz1.to(device, dtype)
            except NameError:
                pass

            # Evaluate loss
            if method_type == "parameterized":
                s_hat, log_r_hat, t_hat0 = model(theta0, x)
                t_hat1 = None
            elif method_type == "doubly_parameterized":
                s_hat, log_r_hat, t_hat0, t_hat1 = model(theta0, theta1, x)
            else:
                raise ValueError("Unknown method type %s", method_type)

            losses = [
                loss_function(s_hat, log_r_hat, t_hat0, t_hat1, y, r_xz, t_xz0, t_xz1)
                for loss_function in loss_functions
            ]
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


def evaluate_ratio_model(
    model,
    method_type=None,
    theta0s=None,
    theta1s=None,
    xs=None,
    evaluate_score=False,
    run_on_gpu=True,
    double_precision=False,
    return_grad_x=False,
):
    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.double if double_precision else torch.float

    # Figure out method type
    if method_type is None:
        if isinstance(model, ParameterizedRatioEstimator):
            method_type = "parameterized"
        elif isinstance(model, DoublyParameterizedRatioEstimator):
            method_type = "doubly_parameterized"
        else:
            raise RuntimeError("Cannot infer method type automatically")

    # Balance theta0 and theta1
    if theta1s is None:
        n_thetas = len(theta0s)
    else:
        n_thetas = max(len(theta0s), len(theta1s))
        if len(theta0s) > len(theta1s):
            theta1s = np.array([theta1s[i % len(theta1s)] for i in range(len(theta0s))])
        elif len(theta0s) < len(theta1s):
            theta0s = np.array([theta0s[i % len(theta0s)] for i in range(len(theta1s))])

    # Prepare data
    n_xs = len(xs)
    theta0s = torch.stack([tensor(theta0s[i % n_thetas], requires_grad=True) for i in range(n_xs)])
    if theta1s is not None:
        theta1s = torch.stack([tensor(theta1s[i % n_thetas], requires_grad=True) for i in range(n_xs)])
    xs = torch.stack([tensor(i) for i in xs])

    model = model.to(device, dtype)
    theta0s = theta0s.to(device, dtype)
    if theta1s is not None:
        theta1s = theta1s.to(device, dtype)
    xs = xs.to(device, dtype)

    # Evaluate ratio estimator with score or x gradients:
    if evaluate_score or return_grad_x:
        model.eval()

        if method_type == "parameterized":
            if return_grad_x:
                s_hat, log_r_hat, t_hat0, x_gradients = model(theta0s, xs, return_grad_x=True)
            else:
                s_hat, log_r_hat, t_hat0 = model(theta0s, xs)
                x_gradients = None
            t_hat1 = None
        elif method_type == "doubly_parameterized":
            if return_grad_x:
                s_hat, log_r_hat, t_hat0, t_hat1, x_gradients = model(theta0s, theta1s, xs, return_grad_x=True)
            else:
                s_hat, log_r_hat, t_hat0, t_hat1 = model(theta0s, theta1s, xs)
                x_gradients = None
        else:
            raise ValueError("Unknown method type %s", method_type)

        # Get data and return
        s_hat = s_hat.detach().numpy().flatten()
        log_r_hat = log_r_hat.detach().numpy().flatten()
        t_hat0 = t_hat0.detach().numpy()
        if t_hat1 is not None:
            t_hat1 = t_hat1.detach().numpy()

    # Evaluate ratio estimator without score:
    else:
        with torch.no_grad():
            model.eval()

            if method_type == "parameterized":
                s_hat, log_r_hat, _ = model(theta0s, xs, track_score=False)
            elif method_type == "doubly_parameterized":
                s_hat, log_r_hat, _, _ = model(theta0s, theta1s, xs, track_score=False)
            else:
                raise ValueError("Unknown method type %s", method_type)

            # Get data and return
            s_hat = s_hat.detach().numpy().flatten()
            log_r_hat = log_r_hat.detach().numpy().flatten()
            t_hat0, t_hat1 = None, None

    if return_grad_x:
        return s_hat, log_r_hat, t_hat0, t_hat1, x_gradients
    return s_hat, log_r_hat, t_hat0, t_hat1
