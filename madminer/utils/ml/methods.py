from __future__ import absolute_import, division, print_function

from madminer.utils.ml import ratio_losses, flow_losses
from madminer.utils.ml.trainer import SingleParameterizedRatioTrainer, DoubleParameterizedRatioTrainer
from madminer.utils.ml.trainer import FlowTrainer, LocalScoreTrainer


def get_trainer(method):
    if method in ["sally", "sallino"]:
        return LocalScoreTrainer
    elif method in ["nde", "scandal"]:
        return FlowTrainer
    elif method in ["carl", "rolr", "rascal", "cascal", "alice", "alices"]:
        return SingleParameterizedRatioTrainer
    elif method in ["carl2", "rolr2", "rascal2", "cascal2", "alice2", "alices2"]:
        return DoubleParameterizedRatioTrainer
    else:
        raise RuntimeError("Unknown method %s", method)


def get_loss(method, alpha):
    if method in ["carl", "carl2"]:
        loss_functions = [ratio_losses.standard_cross_entropy]
        loss_weights = [1.0]
        loss_labels = ["xe"]
    elif method in ["rolr", "rolr2"]:
        loss_functions = [ratio_losses.ratio_mse]
        loss_weights = [1.0]
        loss_labels = ["mse_r"]
    elif method == "rascal":
        loss_functions = [ratio_losses.ratio_mse, ratio_losses.score_mse_num]
        loss_weights = [1.0, alpha]
        loss_labels = ["mse_r", "mse_score"]
    elif method == "rascal2":
        loss_functions = [ratio_losses.ratio_mse, ratio_losses.score_mse]
        loss_weights = [1.0, alpha]
        loss_labels = ["mse_r", "mse_score"]
    elif method in ["alice", "alice2"]:
        loss_functions = [ratio_losses.augmented_cross_entropy]
        loss_weights = [1.0]
        loss_labels = ["improved_xe"]
    elif method == "alices":
        loss_functions = [ratio_losses.augmented_cross_entropy, ratio_losses.score_mse_num]
        loss_weights = [1.0, alpha]
        loss_labels = ["improved_xe", "mse_score"]
    elif method == "alices2":
        loss_functions = [ratio_losses.augmented_cross_entropy, ratio_losses.score_mse]
        loss_weights = [1.0, alpha]
        loss_labels = ["improved_xe", "mse_score"]
    elif method in ["sally", "sallino"]:
        loss_functions = [ratio_losses.local_score_mse]
        loss_weights = [1.0]
        loss_labels = ["mse_score"]
    elif method == "nde":
        loss_functions = [flow_losses.negative_log_likelihood]
        loss_weights = [1.0]
        loss_labels = ["nll"]
    elif method == "scandal":
        loss_functions = [flow_losses.negative_log_likelihood, flow_losses.score_mse]
        loss_weights = [1.0, alpha]
        loss_labels = ["nll", "mse_score"]
    else:
        raise NotImplementedError("Unknown method {}".format(method))
    return loss_functions, loss_labels, loss_weights
