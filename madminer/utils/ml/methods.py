from __future__ import absolute_import, division, print_function

from madminer.utils.ml import ratio_losses, flow_losses
from madminer.utils.ml import ratio_trainer, flow_trainer, score_trainer


def get_training_function(method):
    if method in ["sally", "sallino"]:
        return score_trainer.train_local_score_model
    elif method in ["nde", "scandal"]:
        return flow_trainer.train_flow_model
    else:
        return ratio_trainer.train_ratio_model


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
