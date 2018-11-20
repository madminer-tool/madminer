from __future__ import absolute_import, division, print_function

import six
import logging
import os
import json
import numpy as np

import torch

from madminer.utils.ml import ratio_losses, flow_losses
from madminer.utils.ml.models.maf import ConditionalMaskedAutoregressiveFlow
from madminer.utils.ml.models.maf_mog import ConditionalMixtureMaskedAutoregressiveFlow
from madminer.utils.ml.models.ratio import ParameterizedRatioEstimator, DoublyParameterizedRatioEstimator
from madminer.utils.ml.models.score import LocalScoreEstimator
from madminer.utils.ml.flow_trainer import train_flow_model, evaluate_flow_model
from madminer.utils.ml.ratio_trainer import train_ratio_model, evaluate_ratio_model
from madminer.utils.ml.score_trainer import train_local_score_model, evaluate_local_score_model
from madminer.utils.various import create_missing_folders, load_and_check, general_init, shuffle


class MLForge:
    """
    Estimating likelihood ratios and scores with machine learning.

    Each instance of this class represents one neural estimator. The most important functions are:

    * `MLForge.train()` to train an estimator. The keyword `method` determines the inference technique
      and whether a class instance represents a single-parameterized likelihood ratio estimator, a doubly-parameterized
      likelihood ratio estimator, or a local score estimator.
    * `MLForge.evaluate()` to evaluate the estimator.
    * `MLForge.save()` to save the trained model to files.
    * `MLForge.load()` to load the trained model from files.

    Please see the tutorial for a detailed walk-through.

    Parameters
    ----------
    debug : bool, optional
        If True, additional detailed debugging output is printed. Default value: False.

    """

    def __init__(self, debug=False):
        general_init(debug=debug)

        self.debug = debug

        self.method_type = None
        self.model = None
        self.method = None
        self.nde_type = None
        self.n_observables = None
        self.n_parameters = None
        self.n_hidden = None
        self.activation = None
        self.maf_n_mades = None
        self.maf_batch_norm = None
        self.maf_batch_norm_alpha = None
        self.features = None
        self.x_scaling_means = None
        self.x_scaling_stds = None

    def train(
        self,
        method,
        x_filename,
        y_filename=None,
        theta0_filename=None,
        theta1_filename=None,
        r_xz_filename=None,
        t_xz0_filename=None,
        t_xz1_filename=None,
        features=None,
        nde_type="mafmog",
        n_hidden=(100, 100),
        activation="tanh",
        maf_n_mades=3,
        maf_batch_norm=False,
        maf_batch_norm_alpha=0.1,
        maf_mog_n_components=10,
        alpha=1.0,
        trainer="amsgrad",
        n_epochs=50,
        batch_size=128,
        initial_lr=0.001,
        final_lr=0.0001,
        nesterov_momentum=None,
        validation_split=None,
        early_stopping=True,
        scale_inputs=True,
        shuffle_labels=False,
        grad_x_regularization=None,
    ):

        """
        Trains a neural network to estimate either the likelihood ratio or, if method is 'sally' or 'sallino', the
        score.

        The keyword method determines the structure of the estimator that an instance of this class represents:

        * For 'alice', 'alices', 'carl', 'nde', 'rascal', 'rolr', and 'scandal', the neural network models
          the likelihood ratio as a function of the observables `x` and the numerator hypothesis `theta0`, while
          the denominator hypothesis is kept at a fixed reference value ("single-parameterized likelihood ratio
          estimator"). In addition to the likelihood ratio, the estimator allows to estimate the score at `theta0`.
        * For 'alice2', 'alices2', 'carl2', 'rascal2', and 'rolr2', the neural network models
          the likelihood ratio as a function of the observables `x`, the numerator hypothesis `theta0`, and the
          denominator hypothesis `theta1` ("doubly parameterized likelihood ratio estimator"). The score at `theta0`
          and `theta1` can also be evaluated.
        * For 'sally' and 'sallino', the neural networks models the score evaluated at some reference hypothesis
          ("local score regression"). The likelihood ratio cannot be estimated directly from the neural network, but
          can be estimated in a second step through density estimation in the estimated score space.

        Parameters
        ----------
        method : str
            The inference method used. Allows values are 'alice', 'alices', 'carl', 'nde', 'rascal', 'rolr', and
            'scandal' for a single-parameterized likelihood ratio estimator; 'alice2', 'alices2', 'carl2', 'rascal2',
            and 'rolr2' for a doubly-parameterized likelihood ratio estimator; and 'sally' and 'sallino' for local
            score regression.
            
        x_filename : str
            Path to an unweighted sample of observations, as saved by the `madminer.sampling.SampleAugmenter` functions.
            Required for all inference methods.
            
        y_filename : str or None, optional
            Path to an unweighted sample of class labels, as saved by the `madminer.sampling.SampleAugmenter` functions.
            Required for the 'alice', 'alice2', 'alices', 'alices2', 'carl', 'carl2', 'rascal', 'rascal2', 'rolr',
            and 'rolr2' methods. Default value: None.

        theta0_filename : str or None, optional
            Path to an unweighted sample of numerator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required for the 'alice', 'alice2', 'alices', 'alices2', 'carl', 'carl2', 'nde', 'rascal',
            'rascal2', 'rolr', 'rolr2', and 'scandal' methods. Default value: None.

        theta1_filename : str or None, optional
            Path to an unweighted sample of denominator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required for the 'alice2', 'alices2', 'carl2', 'rascal2', and 'rolr2' methods. Default value:
            None.

        r_xz_filename : str or None, optional
            Path to an unweighted sample of joint likelihood ratios, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required for the 'alice', 'alice2', 'alices', 'alices2', 'rascal', 'rascal2', 'rolr', and 'rolr2'
            methods. Default value: None.

        t_xz0_filename : str or None, optional
            Path to an unweighted sample of joint scores at theta0, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required for the 'alices', 'alices2', 'rascal', 'rascal2', 'sallino', 'sally', and 'scandal'
            methods. Default value: None.

        t_xz1_filename : str or None, optional
            Path to an unweighted sample of joint scores at theta1, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required for the 'rascal2' and 'alices2' methods. Default value: None.

        features : list of int or None, optional
            Indices of observables (features) that are used as input to the neural networks. If None, all observables
            are used. Default value: None.

        nde_type : {'maf', 'mafmog'}, optional
            If the method is 'nde' or 'scandal', nde_type determines the architecture used in the neural density
            estimator. Currently supported are 'maf' for a Masked Autoregressive Flow with a Gaussian base density, or
            'mafmog' for a Masked Autoregressive Flow with a mixture of Gaussian base densities. Default value:
            'mafmog'.

        n_hidden : tuple of int, optional
            Units in each hidden layer in the neural networks. If method is 'nde' or 'scandal', this refers to the
            setup of each individual MADE layer. Default value: (100, 100).
            
        activation : {'tanh', 'sigmoid', 'relu'}, optional
            Activation function. Default value: 'tanh'.

        maf_n_mades : int, optional
            If method is 'nde' or 'scandal', this sets the number of MADE layers. Default value: 3.

        maf_batch_norm : bool, optional
            If method is 'nde' or 'scandal', switches batch normalization layers after each MADE layer on or off.
            Default: False.

        maf_batch_norm_alpha : float, optional
            If method is 'nde' or 'scandal' and maf_batch_norm is True, this sets the alpha parameter in the calculation
            of the running average of the mean and variance. Default value: 0.1.

        maf_mog_n_components : int, optional
            If method is 'nde' or 'scandal' and nde_type is 'mafmog', this sets the number of Gaussian base components.
            Default value: 10.

        alpha : float, optional
            Hyperparameter weighting the score error in the loss function of the 'alices', 'alices2', 'rascal',
            'rascal2', and 'scandal' methods. Default value: 1.

        trainer : {"adam", "amsgrad", "sgd"}, optional
            Optimization algorithm. Default value: "amsgrad".

        n_epochs : int, optional
            Number of epochs. Default value: 50.

        batch_size : int, optional
            Batch size. Default value: 128.

        initial_lr : float, optional
            Learning rate during the first epoch, after which it exponentially decays to final_lr. Default value:
            0.001.

        final_lr : float, optional
            Learning rate during the last epoch. Default value: 0.0001.

        nesterov_momentum : float or None, optional
            If trainer is "sgd", sets the Nesterov momentum. Default value: None.

        validation_split : float or None, optional
            Fraction of samples used  for validation and early stopping (if early_stopping is True). If None, the entire
            sample is used for training and early stopping is deactivated. Default value: None.

        early_stopping : bool, optional
            Activates early stopping based on the validation loss (only if validation_split is not None). Default value:
            True.

        scale_inputs : bool, optional
            Scale the observables to zero mean and unit variance. Default value: True.

        shuffle_labels : bool optional
            If True, the labels (`y`, `r_xz`, `t_xz`) are shuffled, while the observations (`x`) remain in their
            normal order. This serves as a closure test, in particular as cross-check against overfitting: an estimator
            trained with shuffle_labels=True should predict to likelihood ratios around 1 and scores around 0.

        grad_x_regularization : float or None, optional
            If not None, a term of the form `grad_x_regularization * |grad_x f(x)|^2` is added to the loss, where `f(x)`
            is the neural network output (the estimated log likelihood ratio or score). Default value: None.

        Returns
        -------
            None

        """

        logging.info("Starting training")
        logging.info("  Method:                 %s", method)
        logging.info("  Training data: x at %s", x_filename)
        if theta0_filename is not None:
            logging.info("                 theta0 at %s", theta0_filename)
        if theta1_filename is not None:
            logging.info("                 theta1 at %s", theta1_filename)
        if y_filename is not None:
            logging.info("                 y at %s", y_filename)
        if r_xz_filename is not None:
            logging.info("                 r_xz at %s", r_xz_filename)
        if t_xz0_filename is not None:
            logging.info("                 t_xz (theta0) at  %s", t_xz0_filename)
        if t_xz1_filename is not None:
            logging.info("                 t_xz (theta1) at  %s", t_xz1_filename)
        if features is None:
            logging.info("  Features:               all")
        else:
            logging.info("  Features:               %s", features)
        logging.info("  Method:                 %s", method)
        if method in ["nde", "scandal"]:
            logging.info("  Neural density est.:    %s", nde_type)
        if method not in ["nde", "scandal"]:
            logging.info("  Hidden layers:          %s", n_hidden)
        if method in ["nde", "scandal"]:
            logging.info("  MAF, number MADEs:      %s", maf_n_mades)
            logging.info("  MAF, batch norm:        %s", maf_batch_norm)
            logging.info("  MAF, BN alpha:          %s", maf_batch_norm_alpha)
            logging.info("  MAF MoG, components:    %s", maf_mog_n_components)
        logging.info("  Activation function:    %s", activation)
        if method in ["cascal", "cascal2", "rascal", "rascal2", "scandal"]:
            logging.info("  alpha:                  %s", alpha)
        logging.info("  Batch size:             %s", batch_size)
        logging.info("  Trainer:                %s", trainer)
        logging.info("  Epochs:                 %s", n_epochs)
        logging.info("  Learning rate:          %s initially, decaying to %s", initial_lr, final_lr)
        if trainer == "sgd":
            logging.info("  Nesterov momentum:      %s", nesterov_momentum)
        logging.info("  Validation split:       %s", validation_split)
        logging.info("  Early stopping:         %s", early_stopping)
        logging.info("  Scale inputs:           %s", scale_inputs)
        logging.info("  Shuffle labels          %s", shuffle_labels)
        if grad_x_regularization is None:
            logging.info("  Regularization:         None")
        else:
            logging.info("  Regularization:         %s * |grad_x f(x)|^2", grad_x_regularization)

        # Load training data
        logging.info("Loading training data")

        theta0 = load_and_check(theta0_filename)
        theta1 = load_and_check(theta1_filename)
        x = load_and_check(x_filename)
        y = load_and_check(y_filename)
        r_xz = load_and_check(r_xz_filename)
        t_xz0 = load_and_check(t_xz0_filename)
        t_xz1 = load_and_check(t_xz1_filename)

        if y is not None:
            y = y.reshape((-1, 1))

        # Check necessary information is theere
        assert x is not None
        if method in [
            "carl",
            "carl2",
            "nde",
            "scandal",
            "rolr",
            "alice",
            "rascal",
            "alices",
            "rolr2",
            "alice2",
            "rascal2",
            "alices2",
        ]:
            assert theta0 is not None
        if method in ["rolr", "alice", "rascal", "alices", "rolr2", "alice2", "rascal2", "alices2"]:
            assert r_xz is not None
        if method in ["carl", "carl2", "rolr", "alice", "rascal", "alices", "rolr2", "alice2", "rascal2", "alices2"]:
            assert y is not None
        if method in ["scandal", "rascal", "alices", "rascal2", "alices2", "sally", "sallino"]:
            assert t_xz0 is not None
        if method in ["carl2", "rolr2", "alice2", "rascal2", "alices2"]:
            assert theta1 is not None
        if method in ["rascal2", "alices2"]:
            assert t_xz1 is not None

        if method in ["nde", "scandal"]:
            assert nde_type in ["maf", "mafmog"]

        # Infer dimensions of problem
        n_samples = x.shape[0]
        n_observables = x.shape[1]
        if theta0 is not None:
            n_parameters = theta0.shape[1]
        else:
            n_parameters = t_xz0.shape[1]

        logging.info("Found %s samples with %s parameters and %s observables", n_samples, n_parameters, n_observables)

        # Scale features
        if scale_inputs:
            logging.info("Rescaling inputs")
            self.x_scaling_means = np.mean(x, axis=0)
            self.x_scaling_stds = np.maximum(np.std(x, axis=0), 1.0e-6)
            x[:] -= self.x_scaling_means
            x[:] /= self.x_scaling_stds
        else:
            self.x_scaling_means = np.zeros(n_parameters)
            self.x_scaling_stds = np.ones(n_parameters)

        logging.debug("Observable ranges:")
        for i in range(n_observables):
            logging.debug(
                "  x_%s: mean %s, std %s, range %s ... %s",
                i + 1,
                np.mean(x[:, i]),
                np.std(x[:, i]),
                np.min(x[:, i]),
                np.max(x[:, i]),
            )

        # Shuffle labels
        if shuffle_labels:
            logging.info("Shuffling labels")
            y, r_xz, t_xz0, t_xz1 = shuffle(y, r_xz, t_xz0, t_xz1)

        # Features
        if features is not None:
            x = x[:, features]
            logging.info("Only using %s of %s observables", x.shape[1], n_observables)
            n_observables = x.shape[1]

        # Save setup
        self.method = method
        self.n_observables = n_observables
        self.n_parameters = n_parameters
        self.n_hidden = n_hidden
        self.activation = activation
        self.maf_n_mades = maf_n_mades
        self.maf_batch_norm = maf_batch_norm
        self.maf_batch_norm_alpha = maf_batch_norm_alpha
        self.features = features

        # Create model
        logging.info("Creating model for method %s", method)
        if method in ["carl", "rolr", "rascal", "alice", "alices"]:
            self.method_type = "parameterized"
            self.model = ParameterizedRatioEstimator(
                n_observables=n_observables, n_parameters=n_parameters, n_hidden=n_hidden, activation=activation
            )
        elif method in ["carl2", "rolr2", "rascal2", "alice2", "alices2"]:
            self.method_type = "doubly_parameterized"
            self.model = DoublyParameterizedRatioEstimator(
                n_observables=n_observables, n_parameters=n_parameters, n_hidden=n_hidden, activation=activation
            )
        elif method in ["sally", "sallino"]:
            self.method_type = "local_score"
            self.model = LocalScoreEstimator(
                n_observables=n_observables, n_parameters=n_parameters, n_hidden=n_hidden, activation=activation
            )
        elif method in ["nde", "scandal"]:
            self.method_type = "nde"
            if nde_type == "maf":
                self.model = ConditionalMaskedAutoregressiveFlow(
                    n_conditionals=n_parameters,
                    n_inputs=n_observables,
                    n_hiddens=n_hidden,
                    n_mades=maf_n_mades,
                    activation=activation,
                    batch_norm=maf_batch_norm,
                    alpha=maf_batch_norm_alpha,
                )
            elif nde_type == "mafmog":
                self.model = ConditionalMixtureMaskedAutoregressiveFlow(
                    n_conditionals=n_parameters,
                    n_inputs=n_observables,
                    n_components=maf_mog_n_components,
                    n_hiddens=n_hidden,
                    n_mades=maf_n_mades,
                    activation=activation,
                    batch_norm=maf_batch_norm,
                    alpha=maf_batch_norm_alpha,
                )
            else:
                raise RuntimeError("Unknown NDE type {}".format(nde_type))
        else:
            raise RuntimeError("Unknown method {}".format(method))

        # Loss fn
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

        # Train model
        logging.info("Training model")

        if method in ["sally", "sallino"]:
            train_local_score_model(
                model=self.model,
                loss_functions=loss_functions,
                loss_weights=loss_weights,
                loss_labels=loss_labels,
                xs=x,
                t_xzs=t_xz0,
                batch_size=batch_size,
                n_epochs=n_epochs,
                initial_learning_rate=initial_lr,
                final_learning_rate=final_lr,
                validation_split=validation_split,
                early_stopping=early_stopping,
                trainer=trainer,
                nesterov_momentum=nesterov_momentum,
                verbose="all" if self.debug else "some",
                grad_x_regularization=grad_x_regularization,
            )
        elif method in ["nde", "scandal"]:
            train_flow_model(
                model=self.model,
                loss_functions=loss_functions,
                loss_weights=loss_weights,
                loss_labels=loss_labels,
                xs=x,
                theta0s=theta0,
                t_xz0s=t_xz0,
                batch_size=batch_size,
                n_epochs=n_epochs,
                initial_learning_rate=initial_lr,
                final_learning_rate=final_lr,
                validation_split=validation_split,
                early_stopping=early_stopping,
                trainer=trainer,
                nesterov_momentum=nesterov_momentum,
                verbose="all" if self.debug else "some",
                grad_x_regularization=grad_x_regularization,
            )
        else:
            train_ratio_model(
                model=self.model,
                method_type=self.method_type,
                loss_functions=loss_functions,
                loss_weights=loss_weights,
                loss_labels=loss_labels,
                theta0s=theta0,
                theta1s=theta1,
                xs=x,
                ys=y,
                r_xzs=r_xz,
                t_xz0s=t_xz0,
                t_xz1s=t_xz1,
                batch_size=batch_size,
                n_epochs=n_epochs,
                initial_learning_rate=initial_lr,
                final_learning_rate=final_lr,
                validation_split=validation_split,
                early_stopping=early_stopping,
                trainer=trainer,
                nesterov_momentum=nesterov_momentum,
                verbose="all" if self.debug else "some",
                grad_x_regularization=grad_x_regularization,
            )

    def evaluate(
        self,
        x_filename,
        theta0_filename=None,
        theta1_filename=None,
        test_all_combinations=True,
        evaluate_score=False,
        return_grad_x=False,
    ):

        """
        Evaluates a trained estimator of the likelihood ratio (or, if method is 'sally' or 'sallino', the score).

        Parameters
        ----------
        x_filename : str
            Path to an unweighted sample of observations, as saved by the `madminer.sampling.SampleAugmenter` functions.
            
        theta0_filename : str or None, optional
            Path to an unweighted sample of numerator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required if the estimator was trained with the 'alice', 'alice2', 'alices', 'alices2', 'carl',
            'carl2', 'nde', 'rascal', 'rascal2', 'rolr', 'rolr2', or 'scandal' method. Default value: None.

        theta1_filename : str or None, optional
            Path to an unweighted sample of denominator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required if the estimator was trained with the 'alice2', 'alices2', 'carl2', 'rascal2', or
            'rolr2' method. Default value: None.

        test_all_combinations : bool, optional
            If method is not 'sally' and not 'sallino': If False, the number of samples in the observable and theta
            files has to match, and the likelihood ratio is evaluated only for the combinations
            `r(x_i | theta0_i, theta1_i)`. If True, `r(x_i | theta0_j, theta1_j)` for all pairwise combinations `i, j`
            are evaluated. Default value: True.

        evaluate_score : bool, optional
            If method is not 'sally' and not 'sallino', this sets whether in addition to the likelihood ratio the score
            is evaluated. Default value: False.

        return_grad_x : bool, optional
            If True, `grad_x log r(x)` or `grad_x t(x)` (for 'sally' or 'sallino' estimators) are returned in addition
            to the other outputs. Default value: False.

        Returns
        -------
        sally_estimated_score : ndarray
            Only returned if the network was trained with `method='sally'` or `method='sallino'`. In this case, an
            array of the estimator for `t(x_i | theta_ref)` is returned for all events `i`.

        log_likelihood_ratio : ndarray
            Only returned if the network was trained with neither `method='sally'` nor `method='sallino'`. The estimated
            likelihood ratio. If test_all_combinations is True, the result has shape `(n_thetas, n_x)`. Otherwise, it
            has shape `(n_samples,)`.

        score_theta0 : ndarray or None
            Only returned if the network was trained with neither `method='sally'` nor `method='sallino'`. None if
            evaluate_score is False. Otherwise the derived estimated score at `theta0`. If test_all_combinations is
            True, the result has shape `(n_thetas, n_x, n_parameters)`. Otherwise, it has shape
            `(n_samples, n_parameters)`.

        score_theta1 : ndarray or None
            Only returned if the network was trained with neither `method='sally'` nor `method='sallino'`. None if
            evaluate_score is False, or the network was trained with any method other than 'alice2', 'alices2', 'carl2',
            'rascal2', or 'rolr2'. Otherwise the derived estimated score at `theta1`. If test_all_combinations is
            True, the result has shape `(n_thetas, n_x, n_parameters)`. Otherwise, it has shape
            `(n_samples, n_parameters)`.

        grad_x : ndarray
            Only returned if return_grad_x is True.

        """

        if self.model is None:
            raise ValueError("No model -- train or load model before evaluating it!")

        # Load training data
        logging.debug("Loading evaluation data")

        theta0s = load_and_check(theta0_filename)
        theta1s = load_and_check(theta1_filename)
        xs = load_and_check(x_filename)

        # Scale observables
        if self.x_scaling_means is not None and self.x_scaling_stds is not None:
            xs[:] -= self.x_scaling_means
            xs[:] /= self.x_scaling_stds

        # Restrict featuers
        if self.features is not None:
            xs = xs[:, self.features]

        # SALLY evaluation
        if self.method in ["sally", "sallino"]:
            logging.debug("Starting score evaluation")

            if return_grad_x:
                all_t_hat, all_x_gradients = evaluate_local_score_model(model=self.model, xs=xs, return_grad_x=True)

                return all_t_hat, all_x_gradients

            all_t_hat = evaluate_local_score_model(model=self.model, xs=xs)

            return all_t_hat

        # Balance thetas
        if theta1s is None and theta0s is not None:
            theta1s = [None for _ in theta0s]
        elif theta1s is not None and theta0s is not None:
            if len(theta1s) > len(theta0s):
                theta0s = [theta0s[i % len(theta0s)] for i in range(len(theta1s))]
            elif len(theta1s) < len(theta0s):
                theta1s = [theta1s[i % len(theta1s)] for i in range(len(theta0s))]

        # Evaluation for all other methods
        all_log_r_hat = []
        all_t_hat0 = []
        all_t_hat1 = []
        all_x_gradients = []

        if test_all_combinations:
            logging.debug("Starting ratio evaluation for all combinations")

            for i, (theta0, theta1) in enumerate(zip(theta0s, theta1s)):
                logging.debug(
                    "Starting ratio evaluation for thetas %s / %s: %s vs %s", i + 1, len(theta0s), theta0, theta1
                )

                if self.method in ["nde", "scandal"]:
                    _, log_r_hat, t_hat0 = evaluate_flow_model(
                        model=self.model, theta0s=[theta0], xs=xs, evaluate_score=evaluate_score
                    )
                    t_hat1 = None

                else:
                    if return_grad_x:
                        _, log_r_hat, t_hat0, t_hat1, x_gradient = evaluate_ratio_model(
                            model=self.model,
                            method_type=self.method_type,
                            theta0s=[theta0],
                            theta1s=[theta1] if theta1 is not None else None,
                            xs=xs,
                            evaluate_score=evaluate_score,
                            return_grad_x=True,
                        )
                    else:
                        _, log_r_hat, t_hat0, t_hat1 = evaluate_ratio_model(
                            model=self.model,
                            method_type=self.method_type,
                            theta0s=[theta0],
                            theta1s=[theta1] if theta1 is not None else None,
                            xs=xs,
                            evaluate_score=evaluate_score,
                        )
                        x_gradient = None

                all_log_r_hat.append(log_r_hat)
                all_t_hat0.append(t_hat0)
                all_t_hat1.append(t_hat1)
                all_x_gradients.append(x_gradient)

            all_log_r_hat = np.array(all_log_r_hat)
            all_t_hat0 = np.array(all_t_hat0)
            all_t_hat1 = np.array(all_t_hat1)
            all_t_hat1 = np.array(all_t_hat1)
            if return_grad_x:
                all_x_gradients = np.array(all_x_gradients)

        else:
            logging.debug("Starting ratio evaluation")

            if self.method in ["nde", "scandal"]:
                _, all_log_r_hat, t_hat0 = evaluate_flow_model(
                    model=self.model, theta0s=theta0s, xs=xs, evaluate_score=evaluate_score
                )
                all_t_hat1 = None

            else:
                if return_grad_x:
                    _, all_log_r_hat, all_t_hat0, all_t_hat1, all_x_gradients = evaluate_ratio_model(
                        model=self.model,
                        method_type=self.method_type,
                        theta0s=theta0s,
                        theta1s=None if None in theta1s else theta1s,
                        xs=xs,
                        evaluate_score=evaluate_score,
                        return_grad_x=True,
                    )
                else:
                    _, all_log_r_hat, all_t_hat0, all_t_hat1 = evaluate_ratio_model(
                        model=self.model,
                        method_type=self.method_type,
                        theta0s=theta0s,
                        theta1s=None if None in theta1s else theta1s,
                        xs=xs,
                        evaluate_score=evaluate_score,
                    )
                    all_x_gradients = None

        logging.debug("Evaluation done")

        if return_grad_x:
            return all_log_r_hat, all_t_hat0, all_t_hat1, all_x_gradients
        return all_log_r_hat, all_t_hat0, all_t_hat1

    def calculate_fisher_information(self, x, weights=None, n_events=1):

        """
        Calculates the expected Fisher information matrix based on the kinematic information in a given number of
        events. Currently only supported for estimators trained with `method='sally'` or `method='sallino'`.

        Parameters
        ----------
        x : str or ndarray
            Sample of observations, or path to numpy file with observations, as saved by the
            `madminer.sampling.SampleAugmenter` functions. Note that this sample has to be sampled from the reference
            parameter where the score is estimated with the SALLY / SALLINO estimator!

        weights : None or ndarray, optional
            Weights for the observations. If None, all events are taken to have equal weight. Default value: None.
            
        n_events : float, optional
            Expected number of events for which the kinematic Fisher information should be calculated. Default value: 1.

        Returns
        -------
        fisher_information : ndarray
            Expected kinematic Fisher information matrix with shape `(n_parameters, n_parameters)`.

        """

        if self.model is None:
            raise ValueError("No model -- train or load model before evaluating it!")

        # Load training data
        logging.debug("Loading evaluation data")
        if isinstance(x, six.string_types):
            x = load_and_check(x)
        n_samples = x.shape[0]

        # Scale observables
        if self.x_scaling_means is not None and self.x_scaling_stds is not None:
            x[:] -= self.x_scaling_means
            x[:] /= self.x_scaling_stds

        # Restrict featuers
        if self.features is not None:
            x = x[:, self.features]

        # Estimate scores
        if self.method in ["sally", "sallino"]:
            logging.debug("Starting score evaluation")

            t_hats = evaluate_local_score_model(model=self.model, xs=x)
        else:
            raise NotImplementedError("Fisher information calculation only implemented for SALLY estimators")

        # Weights
        if weights is None:
            weights = np.ones(n_samples)
        weights /= np.sum(weights)

        # Calculate Fisher information
        fisher_information = float(n_events) * np.einsum("n,ni,nj->ij", weights, t_hats, t_hats)

        # Calculate expected score
        expected_score = np.mean(t_hats, axis=0)
        logging.debug("Expected per-event score (should be close to zero): %s", expected_score)

        return fisher_information

    def save(self, filename):

        """
        Saves the trained model to four files: a JSON file with the settings, a pickled pyTorch state dict
        file, and numpy files for the mean and variance of the inputs (used for input scaling).

        Parameters
        ----------
        filename : str
            Path to the files. '_settings.json' and '_state_dict.pl' will be added.

        Returns
        -------
            None

        """

        if self.model is None:
            raise ValueError("No model -- train or load model before saving!")

        # Check paths
        create_missing_folders([os.path.dirname(filename)])

        # Save settings
        logging.debug("Saving settings to %s_settings.json", filename)

        settings = {
            "method": self.method,
            "method_type": self.method_type,
            "n_observables": self.n_observables,
            "n_parameters": self.n_parameters,
            "n_hidden": list(self.n_hidden),
            "activation": self.activation,
            "features": self.features,
        }

        with open(filename + "_settings.json", "w") as f:
            json.dump(settings, f)

        # Save scaling
        if self.x_scaling_stds is not None and self.x_scaling_means is not None:
            logging.debug("Saving input scaling information to %s_x_means.npy and %s_x_stds.npy", filename, filename)
            np.save(filename + "_x_means.npy", self.x_scaling_means)
            np.save(filename + "_x_stds.npy", self.x_scaling_stds)

        # Save state dict
        logging.debug("Saving state dictionary to %s_state_dict.pt", filename)
        torch.save(self.model.state_dict(), filename + "_state_dict.pt")

    def load(self, filename):

        """
        Loads a trained model from files.

        Parameters
        ----------
        filename : str
            Path to the files. '_settings.json' and '_state_dict.pl' will be added.

        Returns
        -------
            None

        """

        # Load settings
        logging.debug("Loading settings from %s_settings.json", filename)

        with open(filename + "_settings.json", "r") as f:
            settings = json.load(f)

        self.method = settings["method"]
        self.method_type = settings["method_type"]
        self.n_observables = int(settings["n_observables"])
        self.n_parameters = int(settings["n_parameters"])
        self.n_hidden = tuple([int(item) for item in settings["n_hidden"]])
        self.activation = str(settings["activation"])
        self.features = settings["features"]
        if self.features == "None":
            self.features = None
        if self.features is not None:
            self.features = list([int(item) for item in self.features])

        logging.debug(
            "  Found method %s, %s observables, %s parameters, %s hidden layers, %s activation function, "
            "features %s",
            self.method,
            self.n_observables,
            self.n_parameters,
            self.n_hidden,
            self.activation,
            self.features,
        )

        # Load scaling
        try:
            self.x_scaling_means = np.load(filename + "_x_means.npy")
            self.x_scaling_stds = np.load(filename + "_x_stds.npy")
            logging.debug(
                "  Found input scaling information: means %s, stds %s", self.x_scaling_means, self.x_scaling_stds
            )
        except FileNotFoundError:
            logging.warning("Scaling information not found in %s", filename)
            self.x_scaling_means = None
            self.x_scaling_stds = None

        # Create model
        if self.method in ["carl", "rolr", "rascal", "alice", "alices"]:
            assert self.method_type == "parameterized"
            self.model = ParameterizedRatioEstimator(
                n_observables=self.n_observables,
                n_parameters=self.n_parameters,
                n_hidden=self.n_hidden,
                activation=self.activation,
            )
        elif self.method in ["carl2", "rolr2", "rascal2", "alice2", "alices2"]:
            assert self.method_type == "doubly_parameterized"
            self.model = DoublyParameterizedRatioEstimator(
                n_observables=self.n_observables,
                n_parameters=self.n_parameters,
                n_hidden=self.n_hidden,
                activation=self.activation,
            )
        elif self.method in ["sally", "sallino"]:
            assert self.method_type == "local_score"
            self.model = LocalScoreEstimator(
                n_observables=self.n_observables,
                n_parameters=self.n_parameters,
                n_hidden=self.n_hidden,
                activation=self.activation,
            )
        else:
            raise NotImplementedError("Unknown method {}".format(self.method))

        # Load state dict
        logging.debug("Loading state dictionary from %s_state_dict.pt", filename)
        self.model.load_state_dict(torch.load(filename + "_state_dict.pt"))


class EnsembleForge:
    """
    Ensemble methods for likelihood ratio and score information.

    Generally, EnsembleForge instances can be used very similarly to MLForge instances:

    * The initialization of EnsembleForge takes a list of (trained or untrained) MLForge instances.
    * The methods `EnsembleForge.train_one()` and `EnsembleForge.train_all()` train the estimators (this can also be
      done outside of EnsembleForge).
    * `EnsembleForge.calculate_expectation()` can be used to calculate the expectation of the estimation likelihood
      ratio or the expected estimated score over a validation sample. Ideally (and assuming the correct sampling),
      these expectation values should be close to zero. Deviations from zero therefore point out that the estimator
      is probably inaccurate.
    * `EnsembleForge.evaluate()` and `EnsembleForge.calculate_fisher_information()` can then be used to calculate
      ensemble predictions. The user has the option to treat all estimators equally ('committee method') or to give those
      with expected score / ratio close to zero a higher weight.
    * `EnsembleForge.save()` and `EnsembleForge.load()` can store all estimators in one folder.

    The individual estimators in the ensemble can be trained with different methods, but they have to be of the same
    type: either all estimators are single-parameterized likelihood ratio estimators, or all estimators are
    doubly-parameterized likelihood estimators, or all estimators are local score regressors.

    Parameters
    ----------
    estimators : None or int or list of (MLForge or str), optional
        If int, sets the number of estimators that will be created as new MLForge instances. If list, sets
        the estimators directly, either from MLForge instances or filenames (that are then loaded with
        `MLForge.load()`). If None, the ensemble is initialized without estimators. Note that the estimators have
        to be consistent: either all of them are trained with a local score method ('sally' or 'sallino'); or all of
        them are trained with a single-parameterized method ('carl', 'rolr', 'rascal', 'scandal', 'alice', or 'alices');
        or all of them are trained with a doubly parameterized method ('carl2', 'rolr2', 'rascal2', 'alice2', or
        'alices2'). Mixing estimators of different types within one of these three categories is supported, but mixing
        estimators from different categories is not and will raise a RuntimeException. Default value: None.

    Attributes
    ----------
    estimators : list of MLForge
        The estimators in the form of MLForge instances.

    debug : bool, optional
        If True, additional detailed debugging output is printed. Default value: False.

    """

    def __init__(self, estimators=None, debug=False):
        general_init(debug=debug)
        self.debug = debug

        # Initialize estimators
        if estimators is None:
            self.estimators = []
        elif isinstance(estimators, int):
            self.estimators = [MLForge(debug=debug) for _ in range(estimators)]
        else:
            self.estimators = []
            for estimator in estimators:
                if isinstance(estimator, six.string_types):
                    estimator_object = MLForge(debug=debug)
                    estimator_object.load(estimator)
                elif isinstance(estimator, MLForge):
                    estimator_object = estimator
                else:
                    raise ValueError("Entry {} in estimators is neither str nor MLForge instance")

                self.estimators.append(estimator_object)

        self.n_estimators = len(self.estimators)
        self.expectations = None

        # Consistency checks
        for estimator in self.estimators:
            assert isinstance(estimator, MLForge), "Estimator is no MLForge instance!"

        self._check_consistency()

    def add_estimator(self, estimator):
        """
        Adds an estimator to the ensemble.

        Parameters
        ----------
        estimator : MLForge or str
            The estimator, either as MLForge instance or filename (which is then loaded with `MLForge.load()`).

        Returns
        -------
            None

        """
        if isinstance(estimator, six.string_types):
            estimator_object = MLForge(debug=self.debug)
            estimator_object.load(estimator)
        elif isinstance(estimator, MLForge):
            estimator_object = estimator
        else:
            raise ValueError("Entry {} in estimators is neither str nor MLForge instance")

        self.estimators.append(estimator_object)

    def train_one(self, i, **kwargs):
        """
        Trains an individual estimator. See `MLForge.train()`.

        Parameters
        ----------
        i : int
            The index `0 <= i < n_estimators` of the estimator to be trained.

        kwargs : dict
            Parameters for `MLForge.train()`.

        Returns
        -------
            None

        """

        self._check_consistency(kwargs)

        self.estimators[i].train(**kwargs)

    def train_all(self, **kwargs):
        """
        Trains all estimators. See `MLForge.train()`.

        Parameters
        ----------
        kwargs : dict
            Parameters for `MLForge.train()`. If a value in this dict is a list, it has to have length `n_estimators`
            and contain one value of this parameter for each of the estimators. Otherwise the value is used as parameter
            for the training of all the estimators.

        Returns
        -------
            None

        """
        logging.info("Training %s estimators in ensemble", self.n_estimators)

        for key, value in six.iteritems(kwargs):
            if not isinstance(value, list):
                kwargs[key] = [value for _ in range(self.n_estimators)]

            assert len(kwargs[key]) == self.n_estimators, "Keyword {} has wrong length {}".format(key, len(value))

        self._check_consistency(kwargs)

        for i, estimator in enumerate(self.estimators):
            kwargs_this_estimator = {}
            for key, value in six.iteritems(kwargs):
                kwargs_this_estimator[key] = value[i]

            logging.info("Training estimator %s / %s in ensemble", i + 1, self.n_estimators)
            estimator.train(**kwargs_this_estimator)

    def calculate_expectation(self, x_filename, theta0_filename=None, theta1_filename=None):
        """
        Calculates the expectation of the estimation likelihood ratio or the expected estimated score over a validation
        sample. Ideally (and assuming the correct sampling), these expectation values should be close to zero.
        Deviations from zero therefore point out that the estimator is probably inaccurate.

        Parameters
        ----------
        x_filename : str
            Path to an unweighted sample of observations, as saved by the `madminer.sampling.SampleAugmenter` functions.

        theta0_filename : str or None, optional
            Path to an unweighted sample of numerator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required if the estimators were trained with the 'alice', 'alice2', 'alices', 'alices2', 'carl',
            'carl2', 'nde', 'rascal', 'rascal2', 'rolr', 'rolr2', or 'scandal' method. Default value: None.

        theta1_filename : str or None, optional
            Path to an unweighted sample of denominator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required if the estimators were trained with the 'alice2', 'alices2', 'carl2', 'rascal2', or
            'rolr2' method. Default value: None.

        Returns
        -------
        expectations : ndarray
            Expected score (if the estimators were trained with the 'sally' or 'sallino' methods) or likelihood ratio
            (otherwise).

        """

        logging.info("Calculating expectation for %s estimators in ensemble", self.n_estimators)

        self.expectations = []
        method_type = self._check_consistency()

        for i, estimator in enumerate(self.estimators):
            logging.info("Starting evaluation for estimator %s / %s in ensemble", i + 1, self.n_estimators)

            # Calculate expected score / ratio
            if method_type == "local_score":
                prediction = estimator.evaluate(x_filename, theta0_filename, theta1_filename)
            else:
                raise NotImplementedError("Expectation calculation currently only implemented for SALLY and SALLINO!")

            self.expectations.append(np.mean(prediction, axis=0))

        self.expectations = np.array(self.expectations)

        return self.expectations

    def evaluate(
        self,
        x_filename,
        theta0_filename=None,
        theta1_filename=None,
        test_all_combinations=True,
        vote_expectation_weight=None,
        calculate_covariance=False,
        return_individual_predictions=False,
    ):

        """
        Evaluates the estimators of the likelihood ratio (or, if method is 'sally' or 'sallino', the score), and
        calculates the ensemble mean or variance.

        The user has the option to treat all estimators equally ('committee method') or to give those with expected
        score / ratio close to zero (as calculated by `calculate_expectation()`) a higher weight. In the latter case,
        the ensemble mean `f(x)` is calculated as `f(x)  =  sum_i w_i f_i(x)` with weights
        `w_i  =  exp(-vote_expectation_weight |E[f_i]|) / sum_j exp(-vote_expectation_weight |E[f_j]|)`. Here `f_i(x)`
        are the individual estimators and `E[f_i]` is the expectation value calculated by `calculate_expectation()`.

        Parameters
        ----------
        x_filename : str
            Path to an unweighted sample of observations, as saved by the `madminer.sampling.SampleAugmenter` functions.

        theta0_filename : str or None, optional
            Path to an unweighted sample of numerator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required if the estimator was trained with the 'alice', 'alice2', 'alices', 'alices2', 'carl',
            'carl2', 'nde', 'rascal', 'rascal2', 'rolr', 'rolr2', or 'scandal' method. Default value: None.

        theta1_filename : str or None, optional
            Path to an unweighted sample of denominator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required if the estimator was trained with the 'alice2', 'alices2', 'carl2', 'rascal2', or
            'rolr2' method. Default value: None.

        test_all_combinations : bool, optional
            If method is not 'sally' and not 'sallino': If False, the number of samples in the observable and theta
            files has to match, and the likelihood ratio is evaluated only for the combinations
            `r(x_i | theta0_i, theta1_i)`. If True, `r(x_i | theta0_j, theta1_j)` for all pairwise combinations `i, j`
            are evaluated. Default value: True.

        vote_expectation_weight : float or list of float or None, optional
            Factor that determines how much more weight is given to those estimators with small expectation value (as
            calculated by `calculate_expectation()`). If a list is given, results are returned for each element in the
            list. If None, or if `calculate_expectation()` has not been called, all estimators are treated equal.
            Default value: None.

        calculate_covariance : bool, optional
            Whether the covariance matrix is calculated. Default value: False.

        return_individual_predictions : bool, optional
            Whether the individual estimator predictions are returned. Default value: False.

        Returns
        -------
        mean_prediction : ndarray or list of ndarray
            The (weighted) ensemble mean of the estimators. If the estimators were trained with `method='sally'` or
            `method='sallino'`, this is an array of the estimator for `t(x_i | theta_ref)` for all events `i`.
            Otherwise, the estimated likelihood ratio (if test_all_combinations is True, the result has shape
            `(n_thetas, n_x)`, otherwise, it has shape `(n_samples,)`). If more then one value vote_expectation_weight
            is given, this is a list with results for all entries in vote_expectation_weight.

        covariance : None or ndarray or list of ndarray
            The covariance matrix of the (flattened) predictions, defined as the ensemble covariance. If more then one
            value vote_expectation_weight is given, this is a list with results
            for all entries in vote_expectation_weight. If calculate_covariance is False, None is returned.

        weights : ndarray or list of ndarray
            Only returned if return_individual_predictions is True. The estimator weights `w_i`. If more then one value
            vote_expectation_weight is given, this is a list with results for all entries in vote_expectation_weight.

        individual_predictions : ndarray
            Only returned if return_individual_predictions is True. The individual estimator predictions.

        """
        logging.info("Evaluating %s estimators in ensemble", self.n_estimators)

        # Calculate weights of each estimator in vote
        if self.expectations is None or vote_expectation_weight is None:
            weights = [np.ones(self.n_estimators)]
        else:
            if len(self.expectations.shape) == 1:
                expectations_norm = self.expectations
            elif len(self.expectations.shape) == 2:
                expectations_norm = np.linalg.norm(self.expectations, axis=1)
            else:
                expectations_norm = [np.linalg.norm(expectation) for expectation in self.expectations]

            if not isinstance(vote_expectation_weight, list):
                vote_expectation_weight = [vote_expectation_weight]

            weights = []
            for vote_weight in vote_expectation_weight:
                if vote_weight is None:
                    these_weights = np.ones(self.n_estimators)
                else:
                    these_weights = np.exp(-vote_weight * expectations_norm)
                these_weights /= np.sum(these_weights)
                weights.append(these_weights)

        logging.debug("  Estimator weights: %s", weights)

        # Calculate estimator predictions
        predictions = []
        for i, estimator in enumerate(self.estimators):
            logging.info("Starting evaluation for estimator %s / %s in ensemble", i + 1, self.n_estimators)

            predictions.append(
                estimator.evaluate(
                    x_filename, theta0_filename, theta1_filename, test_all_combinations, evaluate_score=False
                )
            )
        predictions = np.array(predictions)

        # Calculate weighted means and covariance matrices
        means = []
        covariances = []

        for these_weights in weights:
            mean = np.average(predictions, axis=0, weights=these_weights)
            means.append(mean)

            if calculate_covariance:
                predictions_flat = predictions.reshape((predictions.shape[0], -1))

                covariance = np.cov(predictions_flat.T, aweights=these_weights)
            else:
                covariance = None

            covariances.append(covariance)

        # Returns
        if len(weights) == 1:
            if return_individual_predictions:
                return means[0], covariances[0], weights[0], predictions
            return means[0], covariances[0]

        if return_individual_predictions:
            return means, covariances, weights, predictions
        return means, covariances

    def calculate_fisher_information(
        self,
        x,
        obs_weights=None,
        n_events=1,
        uncertainty="ensemble",
        vote_expectation_weight=None,
        return_individual_predictions=False,
    ):
        """
        Calculates the expected Fisher information matrices for each estimator, and then returns the ensemble mean and
        variance.

        The user has the option to treat all estimators equally ('committee method') or to give those with expected
        score / ratio close to zero (as calculated by `calculate_expectation()`) a higher weight. In the latter case,
        the ensemble mean `I` is calculated as `I  =  sum_i w_i I_i` with weights
        `w_i  =  exp(-vote_expectation_weight |E[t_i]|) / sum_j exp(-vote_expectation_weight |E[t_k]|)`. Here `I_i`
        are the individual estimators and `E[t_i]` is the expectation value calculated by `calculate_expectation()`.

        Parameters
        ----------
        x : str
            Sample of observations, or path to numpy file with observations, as saved by the
            `madminer.sampling.SampleAugmenter` functions. Note that this sample has to be sampled from the reference
            parameter where the score is estimated with the SALLY / SALLINO estimator!

        obs_weights : None or ndarray, optional
            Weights for the observations. If None, all events are taken to have equal weight. Default value: None.

        n_events : float, optional
            Expected number of events for which the kinematic Fisher information should be calculated. Default value: 1.

        uncertainty : {"ensemble", "expectation", "sum"}, optional
            How the covariance matrix of the Fisher information estimate is calculate. With "ensemble", the ensemble
            covariance is used. With "expectation", the expectation of the score is used as a measure of the uncertainty
            of the score estimator, and this uncertainty is propagated through to the covariance matrix. With "sum",
            both terms are summed. Default value: "ensemble".

        vote_expectation_weight : float or list of float or None, optional
            Factor that determines how much more weight is given to those estimators with small expectation value (as
            calculated by `calculate_expectation()`). If a list is given, results are returned for each element in the
            list. If None, or if `calculate_expectation()` has not been called, all estimators are treated equal.
            Default value: None.

        return_individual_predictions : bool, optional
            Whether the individual estimator predictions are returned. Default value: False.

        Returns
        -------
        mean_prediction : ndarray or list of ndarray
            The (weighted) ensemble mean of the estimators. If the estimators were trained with `method='sally'` or
            `method='sallino'`, this is an array of the estimator for `t(x_i | theta_ref)` for all events `i`.
            Otherwise, the estimated likelihood ratio (if test_all_combinations is True, the result has shape
            `(n_thetas, n_x)`, otherwise, it has shape `(n_samples,)`). If more then one value vote_expectation_weight
            is given, this is a list with results for all entries in vote_expectation_weight.

        covariance : ndarray or list of ndarray
            The covariance matrix of the Fisher information estimate. Its definition depends on the value of
            uncertainty; by default, the covariance is defined as the ensemble covariance. This object
            has four indices, `cov_(ij)(i'j')`, ordered as i j i' j'. It has shape
            `(n_parameters, n_parameters, n_parameters, n_parameters)`. If more then one value vote_expectation_weight
            is given, this is a list with results for all entries in vote_expectation_weight.

        weights : ndarray or list of ndarray
            Only returned if return_individual_predictions is True. The estimator weights `w_i`. If more then one value
            vote_expectation_weight is given, this is a list with results for all entries in vote_expectation_weight.

        individual_predictions : ndarray
            Only returned if return_individual_predictions is True. The individual estimator predictions.

        """
        logging.debug("Evaluating Fisher information for %s estimators in ensemble", self.n_estimators)

        # Check input

        if uncertainty == "expectation" or uncertainty == "sum":
            if self.expectations is None:
                raise RuntimeError(
                    "Expectations have not been calculated, cannot use uncertainty mode 'expectation' " "or 'sum'!"
                )

        # Calculate estimator_weights of each estimator in vote
        if self.expectations is None or vote_expectation_weight is None:
            estimator_weights = [np.ones(self.n_estimators)]
        else:
            if len(self.expectations.shape) == 1:
                expectations_norm = self.expectations
            elif len(self.expectations.shape) == 2:
                expectations_norm = np.linalg.norm(self.expectations, axis=1)
            else:
                expectations_norm = [np.linalg.norm(expectation) for expectation in self.expectations]

            if not isinstance(vote_expectation_weight, list):
                vote_expectation_weight = [vote_expectation_weight]

            estimator_weights = []
            for vote_weight in vote_expectation_weight:
                if vote_weight is None:
                    these_weights = np.ones(self.n_estimators)
                else:
                    these_weights = np.exp(-vote_weight * expectations_norm)
                these_weights /= np.sum(these_weights)
                estimator_weights.append(these_weights)

        logging.debug("  Estimator estimator_weights: %s", estimator_weights)

        # Calculate estimator predictions
        predictions = []
        for i, estimator in enumerate(self.estimators):
            logging.debug("Starting evaluation for estimator %s / %s in ensemble", i + 1, self.n_estimators)

            predictions.append(estimator.calculate_fisher_information(x=x, weights=obs_weights, n_events=n_events))
        predictions = np.array(predictions)

        # Calculate weighted means and covariance matrices
        means = []
        ensemble_covariances = []

        for these_weights in estimator_weights:
            mean = np.average(predictions, axis=0, weights=these_weights)
            means.append(mean)

            predictions_flat = predictions.reshape((predictions.shape[0], -1))
            covariance = np.cov(predictions_flat.T, aweights=these_weights)
            covariance_shape = (predictions.shape[1], predictions.shape[2], predictions.shape[1], predictions.shape[2])
            covariance = covariance.reshape(covariance_shape)

            ensemble_covariances.append(covariance)

        # Calculate ensemble expectation
        expectation_covariances = None
        if uncertainty == "expectation" or uncertainty == "sum":
            expectation_covariances = []
            for these_weights, expectation in zip(estimator_weights, self.expectations):
                mean_expectation = np.average(expectation, weights=these_weights, axis=0)
                expectation_covariances.append(
                    n_events
                    * np.einsum("a,b,c,d->abcd", mean_expectation, mean_expectation, mean_expectation, mean_expectation)
                )

        # Final covariances
        if uncertainty == "ensemble":
            covariances = ensemble_covariances
        elif uncertainty == "expectation":
            covariances = expectation_covariances
        elif uncertainty == "sum":
            covariances = [cov1 + cov2 for cov1, cov2 in zip(ensemble_covariances, expectation_covariances)]
        else:
            raise ValueError("Unknown uncertainty mode {}".format(uncertainty))

        # Returns
        if len(estimator_weights) == 1:
            if return_individual_predictions:
                return means[0], covariances[0], estimator_weights[0], predictions
            return means[0], covariances[0]

        if return_individual_predictions:
            return means, covariances, estimator_weights, predictions
        return means, covariances

    def save(self, folder):
        """
        Saves the estimator ensemble to a folder.

        Parameters
        ----------
        folder : str
            Path to the folder.

        Returns
        -------
            None

        """

        # Check paths
        create_missing_folders([folder])

        # Save ensemble settings
        logging.debug("Saving ensemble setup to %s/ensemble.json", folder)

        if self.expectations is None:
            expectations = "None"
        else:
            expectations = self.expectations.tolist()

        settings = {"n_estimators": self.n_estimators, "expectations": expectations}

        with open(folder + "/ensemble.json", "w") as f:
            json.dump(settings, f)

        # Save estimators
        for i, estimator in enumerate(self.estimators):
            estimator.save(folder + "/estimator_" + str(i))

    def load(self, folder):
        """
        Loads the estimator ensemble from a folder.

        Parameters
        ----------
        folder : str
            Path to the folder.

        Returns
        -------
            None

        """

        # Load ensemble settings
        logging.debug("Loading ensemble setup from %s/ensemble.json", folder)

        with open(folder + "/ensemble.json", "r") as f:
            settings = json.load(f)

        self.n_estimators = settings["n_estimators"]
        self.expectations = settings["expectations"]
        if self.expectations == "None":
            self.expectations = None
        if self.expectations is not None:
            self.expectations = np.array(self.expectations)

        logging.info("Found ensemble with %s estimators and expectations %s", self.n_estimators, self.expectations)

        # Load estimators
        self.estimators = []
        for i in range(self.n_estimators):
            estimator = MLForge()
            estimator.load(folder + "/estimator_" + str(i))
            self.estimators.append(estimator)

    def _check_consistency(self, keywords=None):
        """
        Internal function that checks if all estimators belong to the same category
        (local score regression, single-parameterized likelihood ratio estimator,
        doubly parameterized likelihood ratio estimator).

        Parameters
        ----------
        keywords : dict or None, optional
            kwargs passed to `train_one()` or `train_all()`.

        Returns
        -------
        method_type : {"local_score", "parameterized", "doubly_parameterized"}
            Method type of this ensemble.

        Raises
        ------
        RuntimeError
            Estimators are inconsistent.

        """
        # Accumulate methods of all estimators
        methods = [estimator.method for estimator in self.estimators]

        if keywords is not None:
            keyword_method = keywords.get("method", None)
            if isinstance(keyword_method, list):
                methods += keyword_method
            else:
                methods.append(keyword_method)

        # Check consistency
        method_type = None
        for method in methods:
            if method in ["sally", "sallino"]:
                this_method_type = "local_score"
            elif method in ["carl", "rolr", "rascal", "alice", "alices", "nde", "scandal"]:
                this_method_type = "parameterized"
            elif method in ["carl2", "rolr2", "rascal2", "alice2", "alices2"]:
                this_method_type = "doubly_parameterized"
            elif method is None:
                continue
            else:
                raise RuntimeError("Unknown method %s", method)

            if method_type is None:
                method_type = this_method_type

            if method_type != this_method_type:
                raise RuntimeError(
                    "Ensemble with inconsistent estimator methods! All methods have to be either"
                    " single-parameterized ratio estimators, doubly parameterized ratio estimators,"
                    " or local score estimators. Found methods " + ", ".join(methods) + "."
                )

        # Return method type of ensemble
        return method_type
