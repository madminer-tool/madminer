from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import logging
import torch
from torch import tensor
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils import clip_grad_norm_

logger = logging.getLogger(__name__)


class EarlyStoppingException(Exception):
    pass


class Trainer:
    def __init__(self, model, run_on_gpu=True, double_precision=False):
        self.model = model
        self.run_on_gpu = run_on_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.run_on_gpu else "cpu")
        self.dtype = torch.double if double_precision else torch.float

        self.model = self.model.to(self.device, self.dtype)

        logger.debug(
            "Training on %s with %s precision",
            "GPU" if self.run_on_gpu else "CPU",
            "double" if double_precision else "single",
        )

    def train(
        self,
        data,
        loss_functions,
        loss_weights=None,
        loss_labels=None,
        epochs=50,
        batch_size=100,
        optimizer=optim.Adam,
        optimizer_kwargs=None,
        initial_lr=0.001,
        final_lr=0.0001,
        validation_split=0.25,
        early_stopping=True,
        early_stopping_patience=None,
        clip_gradient=100.0,
        verbose="some",
        **kwargs
    ):
        logger.debug("Initialising dataset and dataloaders")
        dataset = self.make_dataset(data)
        train_loader, val_loader = self.make_dataloaders(dataset, validation_split, batch_size)

        logger.debug("Setting up optimizer")
        optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        opt = optimizer(self.model.parameters(), lr=initial_lr, **optimizer_kwargs)

        early_stopping = early_stopping and (validation_split is not None) and (epochs > 1)
        best_loss, best_model, best_epoch = None, None, None

        n_losses = len(loss_functions)
        loss_weights = [1.0] * n_losses if loss_weights is None else loss_weights

        # Verbosity
        n_epochs_verbose = None
        if verbose == "all":  # Print output after every epoch
            n_epochs_verbose = 1
        elif verbose == "some":  # Print output after 10%, 20%, ..., 100% progress
            n_epochs_verbose = max(int(round(epochs / 10, 0)), 1)

        logger.debug("Beginning main training loop")
        losses_train, losses_val = [], []

        # Loop over epochs
        for i_epoch in range(epochs):
            lr = self.calculate_lr(i_epoch, epochs, initial_lr, final_lr)
            self.set_lr(optimizer, lr)

            loss_train, loss_val, loss_contributions_train, loss_contributions_val = self.epoch(
                i_epoch, train_loader, val_loader, opt, loss_functions, loss_weights, clip_gradient
            )
            losses_train.append(loss_train)
            losses_val.append(loss_val)

            if early_stopping:
                try:
                    best_loss, best_model, best_epoch = self.check_early_stopping(
                        best_loss, best_model, best_epoch, loss_val, best_epoch
                    )
                except EarlyStoppingException:
                    break

            verbose_epoch = (i_epoch + 1) % n_epochs_verbose == 0
            self.report_epoch(
                i_epoch,
                loss_labels,
                loss_train,
                loss_val,
                loss_contributions_train,
                loss_contributions_val,
                verbose=verbose_epoch,
            )

        self.wrap_up_early_stopping(best_model, losses_val[-1], best_loss, best_epoch)

        logger.debug("Training finished")

        return np.array(losses_train), np.array(losses_val)

    @staticmethod
    def make_dataset(data):
        tensor_data = []
        for value in data.values():
            tensor_data.append(torch.from_numpy(value))
        dataset = TensorDataset(*tensor_data)
        return dataset

    def make_dataloaders(self, dataset, validation_split, batch_size):
        if validation_split is None or validation_split <= 0.0:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=self.run_on_gpu)
            val_loader = None

        else:
            assert 0.0 < validation_split < 1.0, "Wrong validation split: {}".format(validation_split)

            n_samples = len(dataset)
            indices = list(range(n_samples))
            split = int(np.floor(validation_split * n_samples))
            np.random.shuffle(indices)
            train_idx, valid_idx = indices[split:], indices[:split]

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(valid_idx)

            train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, pin_memory=self.run_on_gpu)
            val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=batch_size, pin_memory=self.run_on_gpu)

        return train_loader, val_loader

    @staticmethod
    def calculate_lr(i_epoch, n_epochs, initial_lr, final_lr):
        return initial_lr * (final_lr / initial_lr) ** float(i_epoch / (n_epochs - 1.0))

    @staticmethod
    def set_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def epoch(self, i_epoch, train_loader, val_loader, optimizer, loss_functions, loss_weights, clip_gradient=None):
        n_losses = len(loss_functions)

        self.model.train()
        loss_contributions_train = np.zeros(n_losses)
        loss_train = 0.0

        for i_batch, batch_data in enumerate(train_loader):
            batch_loss, batch_loss_contributions = self.batch_train(
                i_epoch, i_batch, batch_data, loss_functions, loss_weights, optimizer, clip_gradient
            )
            loss_train += batch_loss
            for i, batch_loss_contribution in enumerate(batch_loss_contributions):
                loss_contributions_train[i] += batch_loss_contribution

        loss_contributions_train /= len(train_loader)
        loss_train /= len(train_loader)

        if val_loader is not None:
            self.model.eval()
            loss_contributions_val = np.zeros(n_losses)
            loss_val = 0.0

            for i_batch, batch_data in enumerate(val_loader):
                batch_loss, batch_loss_contributions = self.batch_val(
                    i_epoch, i_batch, batch_data, loss_functions, loss_weights
                )
                loss_val += batch_loss
                for i, batch_loss_contribution in enumerate(batch_loss_contributions):
                    loss_contributions_val[i] += batch_loss_contribution

            loss_contributions_val /= len(val_loader)
            loss_val /= len(val_loader)

        else:
            loss_contributions_val = None
            loss_val = None

        return loss_train, loss_val, loss_contributions_train, loss_contributions_val

    def batch_train(self, i_epoch, i_batch, batch_data, loss_functions, loss_weights, optimizer, clip_gradient=None):
        raise NotImplementedError

    def batch_val(self, i_epoch, i_batch, batch_data, loss_functions, loss_weights):
        raise NotImplementedError

    def check_early_stopping(self, best_loss, best_model, best_epoch, loss, i_epoch):
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_model = self.model.state_dict()
            best_epoch = i_epoch
        return best_loss, best_model, best_epoch

    @staticmethod
    def report_epoch(
        i_epoch, loss_labels, loss_train, loss_val, loss_contributions_train, loss_contributions_val, verbose=False
    ):
        logging_fn = logger.info if verbose else logger.debug

        contr_str_train = ""
        contr_str_val = ""
        for i, (label, value_train, value_val) in enumerate(
            zip(loss_labels, loss_contributions_train, loss_contributions_val)
        ):
            if i > 0:
                contr_str_train += ", "
                contr_str_val += ", "
            contr_str_train += "{}: {:.6f}".format(label, value_train)
            contr_str_val += "{}: {:.6f}".format(label, value_val)
        train_report = "  Epoch {:>3d}: train loss {:6.6f} ({})".format(i_epoch + 1, loss_train, contr_str_train)

        logging_fn(train_report)

        if loss_val is not None:
            contr_str_val = ""
            for i, (label, value_train, value_val) in enumerate(
                zip(loss_labels, loss_contributions_train, loss_contributions_val)
            ):
                if i > 0:
                    contr_str_val += ", "
                contr_str_val += "{}: {:.4f}".format(label, value_val)
            val_report = "            val. loss  {:6.6f} ({})".format(loss_val, contr_str_val)
            logging_fn(val_report)

    def wrap_up_early_stopping(self, best_model, loss_val, best_loss, best_epoch):
        if loss_val < best_loss:
            logger.info(
                "Early stopping after epoch %s, with loss %.2f compared to final loss %.2f",
                best_epoch + 1,
                best_loss,
                loss_val,
            )
            self.model.load_state_dict(best_model)
        else:
            logger.info("Early stopping did not improve performance")


class SingleParameterizedRatioTrainer(Trainer):
    def __init__(self, model, run_on_gpu=True, double_precision=False):
        super(SingleParameterizedRatioTrainer, self).__init__(model, run_on_gpu, double_precision)

        self.data_keys = None
        self.calculate_model_score = True

    def make_dataset(self, data):
        tensor_data = []
        for key, value in zip(data.keys(), data.values()):
            if key == "theta0" and self.calculate_model_score:
                tensor_data.append(torch.tensor(value, requires_grad=True))
            else:
                tensor_data.append(torch.from_numpy(value))
        dataset = TensorDataset(*tensor_data)
        return dataset

    def train(self, data, *args, **kwargs):
        self.data_keys = list(data.keys())
        self.calculate_model_score = "t_xz" in self.data_keys
        if "calculate_model_score" in kwargs.keys():
            self.calculate_model_score = kwargs["calculate_model_score"]

        super(SingleParameterizedRatioTrainer, self).train(data, *args, **kwargs)

    def batch_train(self, i_epoch, i_batch, batch_data, loss_functions, loss_weights, optimizer, clip_gradient):
        theta0 = None
        x = None
        y = None
        r_xz = None
        t_xz = None

        for batch_datum, key in zip(batch_data, self.data_keys):
            if key == "theta0":
                theta0 = batch_datum.to(self.device, self.dtype)
            elif key == "x":
                x = batch_datum.to(self.device, self.dtype)
            elif key == "y":
                y = batch_datum.to(self.device, self.dtype)
            elif key == "r_xz":
                r_xz = batch_datum.to(self.device, self.dtype)
            elif key == "t_xz":
                t_xz = batch_datum.to(self.device, self.dtype)

        optimizer.zero_grad()

        # Forward pass
        s_hat, log_r_hat, t_hat = self.model(theta0, x, track_score=self.calculate_model_score, return_grad_x=False)

        loss_contributions = [
            loss_function(s_hat, log_r_hat, t_hat, None, y, r_xz, t_xz, None) for loss_function in loss_functions
        ]
        loss = loss_weights[0] * loss_contributions[0]
        for _w, _l in zip(loss_weights[1:], loss_contributions[1:]):
            loss += _w * _l

        # Optimizer
        loss.backward()
        if clip_gradient is not None:
            clip_grad_norm_(self.model.parameters(), clip_gradient)

        # Optimizer step
        optimizer.step()

        loss = loss.item()
        loss_contributions = [contrib.item() for contrib in loss_contributions]
        return loss, loss_contributions

    def batch_val(self, i_epoch, i_batch, batch_data, loss_functions, loss_weights):
        theta0 = None
        x = None
        y = None
        r_xz = None
        t_xz = None

        for batch_datum, key in zip(batch_data, self.data_keys):
            if key == "theta0":
                theta0 = batch_datum.to(self.device, self.dtype)
            elif key == "x":
                x = batch_datum.to(self.device, self.dtype)
            elif key == "y":
                y = batch_datum.to(self.device, self.dtype)
            elif key == "r_xz":
                r_xz = batch_datum.to(self.device, self.dtype)
            elif key == "t_xz":
                t_xz = batch_datum.to(self.device, self.dtype)

        # Forward pass
        s_hat, log_r_hat, t_hat = self.model(theta0, x, track_score=self.calculate_model_score, return_grad_x=False)

        loss_contributions = [
            loss_function(s_hat, log_r_hat, t_hat, None, y, r_xz, t_xz, None) for loss_function in loss_functions
        ]
        loss = loss_weights[0] * loss_contributions[0]
        for _w, _l in zip(loss_weights[1:], loss_contributions[1:]):
            loss += _w * _l

        loss = loss.item()
        loss_contributions = [contrib.item() for contrib in loss_contributions]
        return loss, loss_contributions


class DoubleParameterizedRatioTrainer(SingleParameterizedRatioTrainer):
    def make_dataset(self, data):
        tensor_data = []
        for key, value in zip(data.keys(), data.values()):
            if key in ["theta0", "theta1"] and self.calculate_model_score:
                tensor_data.append(torch.tensor(value, requires_grad=True))
            else:
                tensor_data.append(torch.from_numpy(value))
        dataset = TensorDataset(*tensor_data)
        return dataset

    def batch_train(self, i_epoch, i_batch, batch_data, loss_functions, loss_weights, optimizer, clip_gradient):
        theta0 = None
        theta1 = None
        x = None
        y = None
        r_xz = None
        t_xz0 = None
        t_xz1 = None

        for batch_datum, key in zip(batch_data, self.data_keys):
            if key == "theta0":
                theta0 = batch_datum.to(self.device, self.dtype)
            elif key == "theta1":
                theta1 = batch_datum.to(self.device, self.dtype)
            elif key == "x":
                x = batch_datum.to(self.device, self.dtype)
            elif key == "y":
                y = batch_datum.to(self.device, self.dtype)
            elif key == "r_xz":
                r_xz = batch_datum.to(self.device, self.dtype)
            elif key == "t_xz0":
                t_xz0 = batch_datum.to(self.device, self.dtype)
            elif key == "t_xz1":
                t_xz1 = batch_datum.to(self.device, self.dtype)

        optimizer.zero_grad()

        # Forward pass
        s_hat, log_r_hat, t_hat0, t_hat1 = self.model(
            theta0, theta1, x, track_score=self.calculate_model_score, return_grad_x=False
        )

        loss_contributions = [
            loss_function(s_hat, log_r_hat, t_hat0, t_hat1, y, r_xz, t_xz0, t_xz1) for loss_function in loss_functions
        ]
        loss = loss_weights[0] * loss_contributions[0]
        for _w, _l in zip(loss_weights[1:], loss_contributions[1:]):
            loss += _w * _l

        # Optimizer
        loss.backward()
        if clip_gradient is not None:
            clip_grad_norm_(self.model.parameters(), clip_gradient)

        # Optimizer step
        optimizer.step()

        loss = loss.item()
        loss_contributions = [contrib.item() for contrib in loss_contributions]
        return loss, loss_contributions

    def batch_val(self, i_epoch, i_batch, batch_data, loss_functions, loss_weights):
        theta0 = None
        theta1 = None
        x = None
        y = None
        r_xz = None
        t_xz0 = None
        t_xz1 = None

        for batch_datum, key in zip(batch_data, self.data_keys):
            if key == "theta0":
                theta0 = batch_datum.to(self.device, self.dtype)
            elif key == "theta1":
                theta1 = batch_datum.to(self.device, self.dtype)
            elif key == "x":
                x = batch_datum.to(self.device, self.dtype)
            elif key == "y":
                y = batch_datum.to(self.device, self.dtype)
            elif key == "r_xz":
                r_xz = batch_datum.to(self.device, self.dtype)
            elif key == "t_xz0":
                t_xz0 = batch_datum.to(self.device, self.dtype)
            elif key == "t_xz1":
                t_xz1 = batch_datum.to(self.device, self.dtype)

        # Forward pass
        s_hat, log_r_hat, t_hat0, t_hat1 = self.model(
            theta0, theta1, x, track_score=self.calculate_model_score, return_grad_x=False
        )

        loss_contributions = [
            loss_function(s_hat, log_r_hat, t_hat0, t_hat1, y, r_xz, t_xz0, t_xz1) for loss_function in loss_functions
        ]
        loss = loss_weights[0] * loss_contributions[0]
        for _w, _l in zip(loss_weights[1:], loss_contributions[1:]):
            loss += _w * _l

        loss = loss.item()
        loss_contributions = [contrib.item() for contrib in loss_contributions]
        return loss, loss_contributions


class LocalScoreTrainer(Trainer):
    def __init__(self, model, run_on_gpu=True, double_precision=False):
        super(LocalScoreTrainer, self).__init__(model, run_on_gpu, double_precision)
        self.data_keys = None

    def train(self, data, *args, **kwargs):
        data_sorted = {"x": data["x"], "t_xz": data["t_xz"]}
        super(LocalScoreTrainer, self).train(data_sorted, *args, **kwargs)

    def batch_train(self, i_epoch, i_batch, batch_data, loss_functions, loss_weights, optimizer, clip_gradient):
        x = batch_data[0].to(self.device, self.dtype)
        t_xz = batch_data[1].to(self.device, self.dtype)

        optimizer.zero_grad()

        # Forward pass
        t_hat = self.model(x)

        loss_contributions = [loss_function(t_hat, t_xz) for loss_function in loss_functions]
        loss = loss_weights[0] * loss_contributions[0]
        for _w, _l in zip(loss_weights[1:], loss_contributions[1:]):
            loss += _w * _l

        # Optimizer
        loss.backward()
        if clip_gradient is not None:
            clip_grad_norm_(self.model.parameters(), clip_gradient)

        # Optimizer step
        optimizer.step()

        loss = loss.item()
        loss_contributions = [contrib.item() for contrib in loss_contributions]
        return loss, loss_contributions

    def batch_val(self, i_epoch, i_batch, batch_data, loss_functions, loss_weights):
        x = batch_data[0].to(self.device, self.dtype)
        t_xz = batch_data[1].to(self.device, self.dtype)

        # Forward pass
        t_hat = self.model(x)

        loss_contributions = [loss_function(t_hat, t_xz) for loss_function in loss_functions]
        loss = loss_weights[0] * loss_contributions[0]
        for _w, _l in zip(loss_weights[1:], loss_contributions[1:]):
            loss += _w * _l

        loss = loss.item()
        loss_contributions = [contrib.item() for contrib in loss_contributions]
        return loss, loss_contributions


class FlowTrainer(Trainer):
    def __init__(self, model, run_on_gpu=True, double_precision=False):
        super(FlowTrainer, self).__init__(model, run_on_gpu, double_precision)

        self.data_keys = None
        self.calculate_model_score = True

    def make_dataset(self, data):
        tensor_data = []
        for key, value in zip(data.keys(), data.values()):
            if key == "theta" and self.calculate_model_score:
                tensor_data.append(torch.tensor(value, requires_grad=True))
            else:
                tensor_data.append(torch.from_numpy(value))
        dataset = TensorDataset(*tensor_data)
        return dataset

    def train(self, data, *args, **kwargs):
        self.data_keys = list(data.keys())
        self.calculate_model_score = "t_xz" in self.data_keys
        if "calculate_model_score" in kwargs.keys():
            self.calculate_model_score = kwargs["calculate_model_score"]

        super(FlowTrainer, self).train(data, *args, **kwargs)

    def batch_train(self, i_epoch, i_batch, batch_data, loss_functions, loss_weights, optimizer, clip_gradient):
        theta = None
        x = None
        t_xz = None

        for batch_datum, key in zip(batch_data, self.data_keys):
            if key == "theta":
                theta = batch_datum.to(self.device, self.dtype)
            elif key == "x":
                x = batch_datum.to(self.device, self.dtype)
            elif key == "t_xz":
                t_xz = batch_datum.to(self.device, self.dtype)

        optimizer.zero_grad()

        # Forward pass
        if self.calculate_model_score:
            _, log_likelihood, score = self.model.log_likelihood_and_score(theta, x)
        else:
            _, log_likelihood = self.model.log_likelihood(theta, x)
            score = None

        loss_contributions = [loss_function(log_likelihood, score, t_xz) for loss_function in loss_functions]
        loss = loss_weights[0] * loss_contributions[0]
        for _w, _l in zip(loss_weights[1:], loss_contributions[1:]):
            loss += _w * _l

        # Optimizer
        loss.backward()
        if clip_gradient is not None:
            clip_grad_norm_(self.model.parameters(), clip_gradient)

        # Optimizer step
        optimizer.step()

        loss = loss.item()
        loss_contributions = [contrib.item() for contrib in loss_contributions]
        return loss, loss_contributions

    def batch_val(self, i_epoch, i_batch, batch_data, loss_functions, loss_weights):
        theta = None
        x = None
        t_xz = None

        for batch_datum, key in zip(batch_data, self.data_keys):
            if key == "theta":
                theta = batch_datum.to(self.device, self.dtype)
            elif key == "x":
                x = batch_datum.to(self.device, self.dtype)
            elif key == "t_xz":
                t_xz = batch_datum.to(self.device, self.dtype)

        # Forward pass
        if self.calculate_model_score:
            _, log_likelihood, score = self.model.log_likelihood_and_score(theta, x)
        else:
            _, log_likelihood = self.model.log_likelihood(theta, x)
            score = None

        loss_contributions = [loss_function(log_likelihood, score, t_xz) for loss_function in loss_functions]
        loss = loss_weights[0] * loss_contributions[0]
        for _w, _l in zip(loss_weights[1:], loss_contributions[1:]):
            loss += _w * _l

        loss = loss.item()
        loss_contributions = [contrib.item() for contrib in loss_contributions]
        return loss, loss_contributions
