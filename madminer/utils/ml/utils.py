import logging
import numpy as np
import torch

from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset

from madminer.utils.ml import losses


logger = logging.getLogger(__name__)


def get_activation_function(activation):
    if activation == "relu":
        return torch.relu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "sigmoid":
        return torch.relu
    elif activation == "lrelu":
        return F.leaky_relu
    elif activation == "rrelu":
        return torch.rrelu
    elif activation == "prelu":
        return torch.prelu
    elif activation == "elu":
        return F.elu
    elif activation == "selu":
        return torch.selu
    elif activation == "log_sigmoid":
        return F.logsigmoid
    elif activation == "softplus":
        return F.softplus
    else:
        raise ValueError("Activation function %s unknown", activation)


def s_from_r(r):
    return np.clip(1.0 / (1.0 + r), 0.0, 1.0)


def r_from_s(s, epsilon=1.0e-6):
    return np.clip((1.0 - s + epsilon) / (s + epsilon), epsilon, None)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def check_for_nan(label, *tensors):
    for tensor in tensors:
        if tensor is None:
            continue
        if torch.isnan(tensor).any():
            logger.warning("%s contains NaNs!\n%s", label, tensor)
            raise NanException


def check_for_nonpos(label, *tensors):
    for tensor in tensors:
        if tensor is None:
            continue
        if (tensor <= 0.0).any():
            logger.warning("%s contains non-positive numbers!\n%s", label, tensor)
            raise NanException


def check_for_nans_in_parameters(model, check_gradients=True):
    for param in model.parameters():
        if torch.any(torch.isnan(param)):
            return True

        if check_gradients and torch.any(torch.isnan(param.grad)):
            return True

    return False


def get_optimizer(optimizer, nesterov_momentum):
    opt_kwargs = None
    if optimizer == "adam":
        opt = optim.Adam
    elif optimizer == "amsgrad":
        opt = optim.Adam
        opt_kwargs = {"amsgrad": True}
    elif optimizer == "sgd":
        opt = optim.SGD
        if nesterov_momentum is not None:
            opt_kwargs = {"momentum": nesterov_momentum}
    else:
        raise ValueError(f"Unknown optimizer {optimizer}")
    return opt, opt_kwargs


def get_loss(method, alpha):
    if method in ["carl", "carl2"]:
        loss_functions = [losses.ratio_xe]
        loss_weights = [1.0]
        loss_labels = ["xe"]
    elif method in ["rolr", "rolr2"]:
        loss_functions = [losses.ratio_mse]
        loss_weights = [1.0]
        loss_labels = ["mse_r"]
    elif method == "cascal":
        loss_functions = [losses.ratio_xe, losses.ratio_score_mse_num]
        loss_weights = [1.0, alpha]
        loss_labels = ["xe", "mse_score"]
    elif method == "cascal2":
        loss_functions = [losses.ratio_xe, losses.ratio_score_mse]
        loss_weights = [1.0, alpha]
        loss_labels = ["xe", "mse_score"]
    elif method == "rascal":
        loss_functions = [losses.ratio_mse, losses.ratio_score_mse_num]
        loss_weights = [1.0, alpha]
        loss_labels = ["mse_r", "mse_score"]
    elif method == "rascal2":
        loss_functions = [losses.ratio_mse, losses.ratio_score_mse]
        loss_weights = [1.0, alpha]
        loss_labels = ["mse_r", "mse_score"]
    elif method in ["alice", "alice2"]:
        loss_functions = [losses.ratio_augmented_xe]
        loss_weights = [1.0]
        loss_labels = ["improved_xe"]
    elif method == "alices":
        loss_functions = [losses.ratio_augmented_xe, losses.ratio_score_mse_num]
        loss_weights = [1.0, alpha]
        loss_labels = ["improved_xe", "mse_score"]
    elif method == "alices2":
        loss_functions = [losses.ratio_augmented_xe, losses.ratio_score_mse]
        loss_weights = [1.0, alpha]
        loss_labels = ["improved_xe", "mse_score"]
    elif method in ["sally", "sallino"]:
        loss_functions = [losses.local_score_mse]
        loss_weights = [1.0]
        loss_labels = ["mse_score"]
    elif method == "nde":
        loss_functions = [losses.flow_nll]
        loss_weights = [1.0]
        loss_labels = ["nll"]
    elif method == "scandal":
        loss_functions = [losses.flow_nll, losses.flow_score_mse]
        loss_weights = [1.0, alpha]
        loss_labels = ["nll", "mse_score"]
    else:
        raise NotImplementedError(f"Unknown method {method}")

    return loss_functions, loss_labels, loss_weights


class EarlyStoppingException(Exception):
    pass


class NanException(Exception):
    pass


class NumpyDataset(Dataset):
    """ Dataset for numpy arrays with explicit memmap support """

    def __init__(self, *arrays, **kwargs):

        self.dtype = kwargs.get("dtype", torch.float)
        self.memmap = []
        self.data = []
        self.n = None

        for array in arrays:
            if self.n is None:
                self.n = array.shape[0]
            assert array.shape[0] == self.n

            if isinstance(array, np.memmap):
                self.memmap.append(True)
                self.data.append(array)
            else:
                self.memmap.append(False)
                tensor = torch.from_numpy(array).to(self.dtype)
                self.data.append(tensor)

    def __getitem__(self, index):
        items = []
        for memmap, array in zip(self.memmap, self.data):
            if memmap:
                tensor = np.array(array[index])
                items.append(torch.from_numpy(tensor).to(self.dtype))
            else:
                items.append(array[index])
        return tuple(items)

    def __len__(self):
        return self.n
