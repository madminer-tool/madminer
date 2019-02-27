from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import madminer.utils.ml.losses
from madminer.utils.ml import losses
from madminer.utils.ml.trainer import SingleParameterizedRatioTrainer, DoubleParameterizedRatioTrainer
from madminer.utils.ml.trainer import FlowTrainer, LocalScoreTrainer


def get_method_type(method):
    if method in ["carl", "rolr", "cascal", "rascal", "alice", "alices"]:
        method_type = "parameterized"
    elif method in ["carl2", "rolr2", "rascal2", "alice2", "alices2"]:
        method_type = "doubly_parameterized"
    elif method in ["sally", "sallino"]:
        method_type = "local_score"
    elif method in ["nde", "scandal"]:
        method_type = "nde"
    else:
        raise RuntimeError("Unknown method {}".format(method))
    return method_type


def package_training_data(method, x, theta0, theta1, y, r_xz, t_xz0, t_xz1):
    method_type = get_method_type(method)
    data = OrderedDict()
    if method_type == "parameterized":
        data["x"] = x
        data["theta"] = theta0
        data["y"] = y
        if r_xz is not None:
            data["r_xz"] = r_xz
        if t_xz0 is not None:
            data["t_xz"] = t_xz0
    elif method_type == "doubly_parameterized":
        data["x"] = x
        data["theta0"] = theta0
        data["theta1"] = theta1
        data["y"] = y
        if r_xz is not None:
            data["r_xz"] = r_xz
        if t_xz0 is not None:
            data["t_xz0"] = t_xz0
        if t_xz1 is not None:
            data["t_xz1"] = t_xz1
    elif method_type == "local_score":
        data["x"] = x
        data["t_xz"] = t_xz0
    elif method_type == "nde":
        data["x"] = x
        data["theta"] = theta0
        if t_xz0 is not None:
            data["t_xz"] = t_xz0
    return data


def get_trainer(method):
    method_type = get_method_type(method)
    if method_type == "parameterized":
        return SingleParameterizedRatioTrainer
    elif method_type == "doubly_parameterized":
        return DoubleParameterizedRatioTrainer
    elif method_type == "local_score":
        return LocalScoreTrainer
    elif method_type == "nde":
        return FlowTrainer
    else:
        raise RuntimeError("Unknown method %s", method)


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
        loss_functions = [madminer.utils.ml.losses.flow_nll]
        loss_weights = [1.0]
        loss_labels = ["nll"]
    elif method == "scandal":
        loss_functions = [madminer.utils.ml.losses.flow_nll, madminer.utils.ml.losses.flow_score_mse]
        loss_weights = [1.0, alpha]
        loss_labels = ["nll", "mse_score"]
    else:
        raise NotImplementedError("Unknown method {}".format(method))
    return loss_functions, loss_labels, loss_weights
