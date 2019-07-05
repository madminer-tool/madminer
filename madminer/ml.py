from __future__ import absolute_import, division, print_function

import six
import logging
import os
import json
import numpy as np
from collections import OrderedDict
import torch

from madminer.utils.ml.models.maf import ConditionalMaskedAutoregressiveFlow
from madminer.utils.ml.models.maf_mog import ConditionalMixtureMaskedAutoregressiveFlow
from madminer.utils.ml.models.ratio import DenseSingleParameterizedRatioModel, DenseDoublyParameterizedRatioModel
from madminer.utils.ml.models.score import DenseLocalScoreModel
from madminer.utils.ml.eval import evaluate_flow_model, evaluate_ratio_model, evaluate_local_score_model
from madminer.utils.ml.utils import get_optimizer, get_loss
from madminer.utils.various import create_missing_folders, load_and_check, shuffle, restrict_samplesize
from madminer.utils.various import separate_information_blocks
from madminer.utils.ml.trainer import SingleParameterizedRatioTrainer, DoubleParameterizedRatioTrainer
from madminer.utils.ml.trainer import LocalScoreTrainer, FlowTrainer

logger = logging.getLogger(__name__)


class Estimator(object):
    """
    Abstract class for any ML estimator. Subclassed by ParameterizedRatioEstimator, DoubleParameterizedRatioEstimator,
    ScoreEstimator, and LikelihoodEstimator.

    Each instance of this class represents one neural estimator. The most important functions are:

    * `Estimator.train()` to train an estimator. The keyword `method` determines the inference technique
      and whether a class instance represents a single-parameterized likelihood ratio estimator, a doubly-parameterized
      likelihood ratio estimator, or a local score estimator.
    * `Estimator.evaluate()` to evaluate the estimator.
    * `Estimator.save()` to save the trained model to files.
    * `Estimator.load()` to load the trained model from files.

    Please see the tutorial for a detailed walk-through.
    """

    def __init__(self, features=None, n_hidden=(100,), activation="tanh"):
        self.features = features
        self.n_hidden = n_hidden
        self.activation = activation

        self.model = None
        self.n_observables = None
        self.n_parameters = None
        self.x_scaling_means = None
        self.x_scaling_stds = None

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate_log_likelihood(self, *args, **kwargs):
        """
        Log likelihood estimation. Signature depends on the type of estimator. The first returned value is the log
        likelihood with shape `(n_thetas, n_x)`.
        """
        raise NotImplementedError

    def evaluate_log_likelihood_ratio(self, *args, **kwargs):
        """
        Log likelihood ratio estimation. Signature depends on the type of estimator. The first returned value is the log
        likelihood ratio with shape `(n_thetas, n_x)` or `(n_x)`.
        """
        raise NotImplementedError

    def evaluate_score(self, *args, **kwargs):
        """
        Score estimation. Signature depends on the type of estimator. The only returned value is the score with shape
        `(n_x)`.
        """
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def calculate_fisher_information(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, filename, save_model=False):

        """
        Saves the trained model to four files: a JSON file with the settings, a pickled pyTorch state dict
        file, and numpy files for the mean and variance of the inputs (used for input scaling).

        Parameters
        ----------
        filename : str
            Path to the files. '_settings.json' and '_state_dict.pl' will be added.

        save_model : bool, optional
            If True, the whole model is saved in addition to the state dict. This is not necessary for loading it
            again with Estimator.load(), but can be useful for debugging, for instance to plot the computational graph.

        Returns
        -------
            None

        """

        logger.info("Saving model to %s", filename)

        if self.model is None:
            raise ValueError("No model -- train or load model before saving!")

        # Check paths
        create_missing_folders([os.path.dirname(filename)])

        # Save settings
        logger.debug("Saving settings to %s_settings.json", filename)

        settings = self._wrap_settings()

        with open(filename + "_settings.json", "w") as f:
            json.dump(settings, f)

        # Save scaling
        if self.x_scaling_stds is not None and self.x_scaling_means is not None:
            logger.debug("Saving input scaling information to %s_x_means.npy and %s_x_stds.npy", filename, filename)
            np.save(filename + "_x_means.npy", self.x_scaling_means)
            np.save(filename + "_x_stds.npy", self.x_scaling_stds)

        # Save state dict
        logger.debug("Saving state dictionary to %s_state_dict.pt", filename)
        torch.save(self.model.state_dict(), filename + "_state_dict.pt")

        # Save model
        if save_model:
            logger.debug("Saving model to %s_model.pt", filename)
            torch.save(self.model, filename + "_model.pt")

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

        logger.info("Loading model from %s", filename)

        # Load settings and create model
        logger.debug("Loading settings from %s_settings.json", filename)
        with open(filename + "_settings.json", "r") as f:
            settings = json.load(f)
        self._unwrap_settings(settings)
        self._create_model()

        # Load scaling
        try:
            self.x_scaling_means = np.load(filename + "_x_means.npy")
            self.x_scaling_stds = np.load(filename + "_x_stds.npy")
            logger.debug(
                "  Found input scaling information: means %s, stds %s", self.x_scaling_means, self.x_scaling_stds
            )
        except FileNotFoundError:
            logger.warning("Scaling information not found in %s", filename)
            self.x_scaling_means = None
            self.x_scaling_stds = None

        # Load state dict
        logger.debug("Loading state dictionary from %s_state_dict.pt", filename)
        self.model.load_state_dict(torch.load(filename + "_state_dict.pt", map_location="cpu"))

    def _initialize_input_transform(self, x, transform=True):
        if transform:
            self.x_scaling_means = np.mean(x, axis=0)
            self.x_scaling_stds = np.maximum(np.std(x, axis=0), 1.0e-6)
        else:
            n_parameters = x.shape[0]

            self.x_scaling_means = np.zeros(n_parameters)
            self.x_scaling_stds = np.ones(n_parameters)

    def _transform_inputs(self, x):
        if self.x_scaling_means is not None and self.x_scaling_stds is not None:
            x_scaled = x - self.x_scaling_means
            x_scaled /= self.x_scaling_stds
        else:
            x_scaled = x
        return x_scaled

    def _wrap_settings(self):
        settings = {
            "n_observables": self.n_observables,
            "n_parameters": self.n_parameters,
            "features": self.features,
            "n_hidden": list(self.n_hidden),
            "activation": self.activation,
        }
        return settings

    def _unwrap_settings(self, settings):
        try:
            _ = str(settings["estimator_type"])
        except KeyError:
            raise RuntimeError(
                "Can't find estimator type information in file. Maybe this file was created with"
                " an incompatible MadMiner version < v0.3.0?"
            )

        self.n_observables = int(settings["n_observables"])
        self.n_parameters = int(settings["n_parameters"])
        self.n_hidden = tuple([int(item) for item in settings["n_hidden"]])
        self.activation = str(settings["activation"])
        self.features = settings["features"]
        if self.features == "None":
            self.features = None
        if self.features is not None:
            self.features = list([int(item) for item in self.features])

    def _create_model(self):
        raise NotImplementedError


class ParameterizedRatioEstimator(Estimator):
    """
    A neural estimator of the likelihood ratio as a function of the observation x as well as
    the numerator hypothesis theta. The reference (denominator) hypothesis is kept fixed at some
    reference value and NOT modeled by the network.

    Parameters
    ----------
    features : list of int or None, optional
        Indices of observables (features) that are used as input to the neural networks. If None, all observables
        are used. Default value: None.

    n_hidden : tuple of int, optional
        Units in each hidden layer in the neural networks. If method is 'nde' or 'scandal', this refers to the
        setup of each individual MADE layer. Default value: (100,).

    activation : {'tanh', 'sigmoid', 'relu'}, optional
        Activation function. Default value: 'tanh'.


    """

    def train(
        self,
        method,
        x,
        y,
        theta,
        r_xz=None,
        t_xz=None,
        alpha=1.0,
        optimizer="amsgrad",
        n_epochs=50,
        batch_size=128,
        initial_lr=0.001,
        final_lr=0.0001,
        nesterov_momentum=None,
        validation_split=0.25,
        early_stopping=True,
        scale_inputs=True,
        shuffle_labels=False,
        limit_samplesize=None,
        memmap=False,
        verbose="some",
    ):

        """
        Trains the network.

        Parameters
        ----------
        method : str
            The inference method used for training. Allowed values are 'alice', 'alices', 'carl', 'cascal', 'rascal',
            and 'rolr'.

        x : ndarray or str
            Path to an unweighted sample of observations, as saved by the `madminer.sampling.SampleAugmenter` functions.
            Required for all inference methods.

        y : ndarray or str
            Class labels (0 = numeerator, 1 = denominator), or filename of a pickled numpy array.

        theta : ndarray or str
            Numerator parameter point, or filename of a pickled numpy array.

        r_xz : ndarray or str or None, optional
            Joint likelihood ratio, or filename of a pickled numpy array. Default value: None.

        t_xz : ndarray or str or None, optional
            Joint scores at theta, or filename of a pickled numpy array. Default value: None.

        alpha : float, optional
            Hyperparameter weighting the score error in the loss function of the 'alices', 'rascal', and 'cascal'
            methods. Default value: 1.

        optimizer : {"adam", "amsgrad", "sgd"}, optional
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
            sample is used for training and early stopping is deactivated. Default value: 0.25.

        early_stopping : bool, optional
            Activates early stopping based on the validation loss (only if validation_split is not None). Default value:
            True.

        scale_inputs : bool, optional
            Scale the observables to zero mean and unit variance. Default value: True.

        shuffle_labels : bool, optional
            If True, the labels (`y`, `r_xz`, `t_xz`) are shuffled, while the observations (`x`) remain in their
            normal order. This serves as a closure test, in particular as cross-check against overfitting: an estimator
            trained with shuffle_labels=True should predict to likelihood ratios around 1 and scores around 0.

        limit_samplesize : int or None, optional
            If not None, only this number of samples (events) is used to train the estimator. Default value: None.

        memmap : bool, optional.
            If True, training files larger than 1 GB will not be loaded into memory at once. Default value: False.

        verbose : {"all", "many", "some", "few", "none}, optional
            Determines verbosity of training. Default value: "some".

        Returns
        -------
            None

        """

        logger.info("Starting training")
        logger.info("  Method:                 %s", method)
        if method in ["cascal", "rascal", "alices"]:
            logger.info("  alpha:                  %s", alpha)
        logger.info("  Batch size:             %s", batch_size)
        logger.info("  Optimizer:              %s", optimizer)
        logger.info("  Epochs:                 %s", n_epochs)
        logger.info("  Learning rate:          %s initially, decaying to %s", initial_lr, final_lr)
        if optimizer == "sgd":
            logger.info("  Nesterov momentum:      %s", nesterov_momentum)
        logger.info("  Validation split:       %s", validation_split)
        logger.info("  Early stopping:         %s", early_stopping)
        logger.info("  Scale inputs:           %s", scale_inputs)
        logger.info("  Shuffle labels          %s", shuffle_labels)
        if limit_samplesize is None:
            logger.info("  Samples:                all")
        else:
            logger.info("  Samples:                %s", limit_samplesize)

        # Load training data
        logger.info("Loading training data")
        memmap_threshold = 1.0 if memmap else None
        theta = load_and_check(theta, memmap_files_larger_than_gb=memmap_threshold)
        x = load_and_check(x, memmap_files_larger_than_gb=memmap_threshold)
        y = load_and_check(y, memmap_files_larger_than_gb=memmap_threshold)
        r_xz = load_and_check(r_xz, memmap_files_larger_than_gb=memmap_threshold)
        t_xz = load_and_check(t_xz, memmap_files_larger_than_gb=memmap_threshold)

        self._check_required_data(method, r_xz, t_xz)

        # Infer dimensions of problem
        n_samples = x.shape[0]
        n_observables = x.shape[1]
        n_parameters = theta.shape[1]
        logger.info("Found %s samples with %s parameters and %s observables", n_samples, n_parameters, n_observables)

        # Limit sample size
        if limit_samplesize is not None and limit_samplesize < n_samples:
            logger.info("Only using %s of %s training samples", limit_samplesize, n_samples)
            x, theta, y, r_xz, t_xz = restrict_samplesize(limit_samplesize, x, theta, y, r_xz, t_xz)

        # Scale features
        if scale_inputs:
            logger.info("Rescaling inputs")
            self._initialize_input_transform(x)
            x = self._transform_inputs(x)
        else:
            self._initialize_input_transform(x, False)

        # Shuffle labels
        if shuffle_labels:
            logger.info("Shuffling labels")
            y, r_xz, t_xz = shuffle(y, r_xz, t_xz)

        # Features
        if self.features is not None:
            x = x[:, self.features]
            logger.info("Only using %s of %s observables", x.shape[1], n_observables)
            n_observables = x.shape[1]

        # Check consistency of input with model
        if self.n_observables is None:
            self.n_observables = n_observables
        if self.n_parameters is None:
            self.n_parameters = n_parameters

        if n_parameters != self.n_parameters:
            raise RuntimeError(
                "Number of parameters does not match model: {} vs {}".format(n_parameters, self.n_parameters)
            )
        if n_observables != self.n_observables:
            raise RuntimeError(
                "Number of observables does not match model: {} vs {}".format(n_observables, self.n_observables)
            )

        # Data
        data = self._package_training_data(method, x, theta, y, r_xz, t_xz)

        # Create model
        if self.model is None:
            logger.info("Creating model")
            self._create_model()

        # Losses
        loss_functions, loss_labels, loss_weights = get_loss(method, alpha)

        # Optimizer
        opt, opt_kwargs = get_optimizer(optimizer, nesterov_momentum)

        # Train model
        logger.info("Training model")
        trainer = SingleParameterizedRatioTrainer(self.model)
        result = trainer.train(
            data=data,
            loss_functions=loss_functions,
            loss_weights=loss_weights,
            loss_labels=loss_labels,
            epochs=n_epochs,
            batch_size=batch_size,
            optimizer=opt,
            optimizer_kwargs=opt_kwargs,
            initial_lr=initial_lr,
            final_lr=final_lr,
            validation_split=validation_split,
            early_stopping=early_stopping,
            verbose=verbose,
        )
        return result

    def evaluate_log_likelihood_ratio(self, x, theta, test_all_combinations=True, evaluate_score=False):
        """
        Evaluates the log likelihood ratio for given observations x betwen the given parameter point theta and the
        reference hypothesis.

        Parameters
        ----------
        x : str or ndarray
            Observations or filename of a pickled numpy array.

        theta : ndarray or str
            Parameter points or filename of a pickled numpy array.

        test_all_combinations : bool, optional
            If False, the number of samples in the observable and theta
            files has to match, and the likelihood ratio is evaluated only for the combinations
            `r(x_i | theta0_i, theta1_i)`. If True, `r(x_i | theta0_j, theta1_j)` for all pairwise combinations `i, j`
            are evaluated. Default value: True.

        evaluate_score : bool, optional
            Sets whether in addition to the likelihood ratio the score is evaluated. Default value: False.

        Returns
        -------
        log_likelihood_ratio : ndarray
            The estimated log likelihood ratio. If test_all_combinations is True, the result has shape
            `(n_thetas, n_x)`. Otherwise, it has shape `(n_samples,)`.

        score : ndarray or None
            None if evaluate_score is False. Otherwise the derived estimated score at `theta0`. If test_all_combinations
            is True, the result has shape `(n_thetas, n_x, n_parameters)`. Otherwise, it has shape
            `(n_samples, n_parameters)`.

        """
        if self.model is None:
            raise ValueError("No model -- train or load model before evaluating it!")

        # Load training data
        logger.info("Loading evaluation data")
        x = load_and_check(x)
        theta = load_and_check(theta)

        # Scale observables
        x = self._transform_inputs(x)

        # Restrict features
        if self.features is not None:
            x = x[:, self.features]

        all_log_r_hat = []
        all_t_hat = []

        if test_all_combinations:
            logger.info("Starting ratio evaluation for %s x-theta combinations", len(theta) * len(x))

            for i, this_theta in enumerate(theta):
                logger.debug("Starting ratio evaluation for thetas %s / %s: %s", i + 1, len(theta), this_theta)
                _, log_r_hat, t_hat, _ = evaluate_ratio_model(
                    model=self.model,
                    method_type="parameterized_ratio",
                    theta0s=[this_theta],
                    theta1s=None,
                    xs=x,
                    evaluate_score=evaluate_score,
                )

                all_log_r_hat.append(log_r_hat)
                all_t_hat.append(t_hat)

            all_log_r_hat = np.array(all_log_r_hat)
            all_t_hat = np.array(all_t_hat)

        else:
            logger.info("Starting ratio evaluation")
            _, all_log_r_hat, all_t_hat, _ = evaluate_ratio_model(
                model=self.model,
                method_type="parameterized_ratio",
                theta0s=theta,
                theta1s=None,
                xs=x,
                evaluate_score=evaluate_score,
            )

        logger.info("Evaluation done")
        return all_log_r_hat, all_t_hat

    def evaluate_log_likelihood(self, *args, **kwargs):
        raise TheresAGoodReasonThisDoesntWork(
            "This estimator can only estimate likelihood ratios, not the likelihood " "itself!"
        )

    def evaluate_score(self, *args, **kwargs):
        raise NotImplementedError("Please use evaluate_log_likelihood_ratio(evaluate_score=True).")

    def calculate_fisher_information(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use evaluate_log_likelihood_ratio(evaluate_score=True) and calculate the "
            "Fisher information manually."
        )

    def evaluate(self, *args, **kwargs):
        return self.evaluate_log_likelihood_ratio(*args, **kwargs)

    def _create_model(self):
        self.model = DenseSingleParameterizedRatioModel(
            n_observables=self.n_observables,
            n_parameters=self.n_parameters,
            n_hidden=self.n_hidden,
            activation=self.activation,
        )

    @staticmethod
    def _check_required_data(method, r_xz, t_xz):
        if method in ["cascal", "alices", "rascal"] and t_xz is None:
            raise RuntimeError("Method {} requires joint score information".format(method))
        if method in ["rolr", "alices", "rascal"] and r_xz is None:
            raise RuntimeError("Method {} requires joint likelihood ratio information".format(method))

    @staticmethod
    def _package_training_data(method, x, theta, y, r_xz, t_xz):
        data = OrderedDict()
        data["x"] = x
        data["theta"] = theta
        data["y"] = y
        if method in ["rolr", "alice", "alices", "rascal"]:
            data["r_xz"] = r_xz
        if method in ["cascal", "alices", "rascal"]:
            data["t_xz"] = t_xz
        return data

    def _wrap_settings(self):
        settings = super(ParameterizedRatioEstimator, self)._wrap_settings()
        settings["estimator_type"] = "parameterized_ratio"
        return settings

    def _unwrap_settings(self, settings):
        super(ParameterizedRatioEstimator, self)._unwrap_settings(settings)

        estimator_type = str(settings["estimator_type"])
        if estimator_type != "parameterized_ratio":
            raise RuntimeError("Saved model is an incompatible estimator type {}.".format(estimator_type))


class DoubleParameterizedRatioEstimator(Estimator):
    """
    A neural estimator of the likelihood ratio as a function of the observation x, the numerator hypothesis theta0, and
    the denominator hypothesis theta1.

    Parameters
    ----------
    features : list of int or None, optional
        Indices of observables (features) that are used as input to the neural networks. If None, all observables
        are used. Default value: None.

    n_hidden : tuple of int, optional
        Units in each hidden layer in the neural networks. If method is 'nde' or 'scandal', this refers to the
        setup of each individual MADE layer. Default value: (100,).

    activation : {'tanh', 'sigmoid', 'relu'}, optional
        Activation function. Default value: 'tanh'.


    """

    def train(
        self,
        method,
        x,
        y,
        theta0,
        theta1,
        r_xz=None,
        t_xz0=None,
        t_xz1=None,
        alpha=1.0,
        optimizer="amsgrad",
        n_epochs=50,
        batch_size=128,
        initial_lr=0.001,
        final_lr=0.0001,
        nesterov_momentum=None,
        validation_split=0.25,
        early_stopping=True,
        scale_inputs=True,
        shuffle_labels=False,
        limit_samplesize=None,
        memmap=False,
        verbose="some",
    ):

        """
        Trains the network.

        Parameters
        ----------
        method : str
            The inference method used for training. Allowed values are 'alice', 'alices', 'carl', 'cascal', 'rascal',
            and 'rolr'.

        x : ndarray or str
            Path to an unweighted sample of observations, as saved by the `madminer.sampling.SampleAugmenter` functions.
            Required for all inference methods.

        y : ndarray or str
            Class labels (0 = numeerator, 1 = denominator), or filename of a pickled numpy array.

        theta0 : ndarray or str
            Numerator parameter point, or filename of a pickled numpy array.

        theta1 : ndarray or str
            Denominator parameter point, or filename of a pickled numpy array.

        r_xz : ndarray or str or None, optional
            Joint likelihood ratio, or filename of a pickled numpy array. Default value: None.

        t_xz0 : ndarray or str or None, optional
            Joint scores at theta0, or filename of a pickled numpy array. Default value: None.

        t_xz1 : ndarray or str or None, optional
            Joint scores at theta1, or filename of a pickled numpy array. Default value: None.

        alpha : float, optional
            Hyperparameter weighting the score error in the loss function of the 'alices', 'rascal', and 'cascal'
            methods. Default value: 1.

        optimizer : {"adam", "amsgrad", "sgd"}, optional
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
            sample is used for training and early stopping is deactivated. Default value: 0.25.

        early_stopping : bool, optional
            Activates early stopping based on the validation loss (only if validation_split is not None). Default value:
            True.

        scale_inputs : bool, optional
            Scale the observables to zero mean and unit variance. Default value: True.

        shuffle_labels : bool, optional
            If True, the labels (`y`, `r_xz`, `t_xz`) are shuffled, while the observations (`x`) remain in their
            normal order. This serves as a closure test, in particular as cross-check against overfitting: an estimator
            trained with shuffle_labels=True should predict to likelihood ratios around 1 and scores around 0.

        limit_samplesize : int or None, optional
            If not None, only this number of samples (events) is used to train the estimator. Default value: None.

        memmap : bool, optional.
            If True, training files larger than 1 GB will not be loaded into memory at once. Default value: False.

        verbose : {"all", "many", "some", "few", "none}, optional
            Determines verbosity of training. Default value: "some".

        Returns
        -------
            None

        """

        logger.info("Starting training")
        logger.info("  Method:                 %s", method)
        if method in ["cascal", "rascal", "alices"]:
            logger.info("  alpha:                  %s", alpha)
        logger.info("  Batch size:             %s", batch_size)
        logger.info("  Optimizer:              %s", optimizer)
        logger.info("  Epochs:                 %s", n_epochs)
        logger.info("  Learning rate:          %s initially, decaying to %s", initial_lr, final_lr)
        if optimizer == "sgd":
            logger.info("  Nesterov momentum:      %s", nesterov_momentum)
        logger.info("  Validation split:       %s", validation_split)
        logger.info("  Early stopping:         %s", early_stopping)
        logger.info("  Scale inputs:           %s", scale_inputs)
        logger.info("  Shuffle labels          %s", shuffle_labels)
        if limit_samplesize is None:
            logger.info("  Samples:                all")
        else:
            logger.info("  Samples:                %s", limit_samplesize)

        # Load training data
        logger.info("Loading training data")
        memmap_threshold = 1.0 if memmap else None
        theta0 = load_and_check(theta0, memmap_files_larger_than_gb=memmap_threshold)
        theta1 = load_and_check(theta1, memmap_files_larger_than_gb=memmap_threshold)
        x = load_and_check(x, memmap_files_larger_than_gb=memmap_threshold)
        y = load_and_check(y, memmap_files_larger_than_gb=memmap_threshold)
        r_xz = load_and_check(r_xz, memmap_files_larger_than_gb=memmap_threshold)
        t_xz0 = load_and_check(t_xz0, memmap_files_larger_than_gb=memmap_threshold)
        t_xz1 = load_and_check(t_xz1, memmap_files_larger_than_gb=memmap_threshold)

        self._check_required_data(method, r_xz, t_xz0, t_xz1)

        # Infer dimensions of problem
        n_samples = x.shape[0]
        n_observables = x.shape[1]
        n_parameters = theta0.shape[1]
        logger.info("Found %s samples with %s parameters and %s observables", n_samples, n_parameters, n_observables)

        # Limit sample size
        if limit_samplesize is not None and limit_samplesize < n_samples:
            logger.info("Only using %s of %s training samples", limit_samplesize, n_samples)
            x, theta0, theta1, y, r_xz, t_xz0, t_xz1 = restrict_samplesize(
                limit_samplesize, x, theta0, theta1, y, r_xz, t_xz0, t_xz1
            )

        # Scale features
        if scale_inputs:
            logger.info("Rescaling inputs")
            self._initialize_input_transform(x)
            x = self._transform_inputs(x)
        else:
            self._initialize_input_transform(x, False)

        # Shuffle labels
        if shuffle_labels:
            logger.info("Shuffling labels")
            y, r_xz, t_xz0, t_xz1 = shuffle(y, r_xz, t_xz0, t_xz1)

        # Features
        if self.features is not None:
            x = x[:, self.features]
            logger.info("Only using %s of %s observables", x.shape[1], n_observables)
            n_observables = x.shape[1]

        # Check consistency of input with model
        if self.n_observables is None:
            self.n_observables = n_observables
        if self.n_parameters is None:
            self.n_parameters = n_parameters

        if n_parameters != self.n_parameters:
            raise RuntimeError(
                "Number of parameters does not match model: {} vs {}".format(n_parameters, self.n_parameters)
            )
        if n_observables != self.n_observables:
            raise RuntimeError(
                "Number of observables does not match model: {} vs {}".format(n_observables, self.n_observables)
            )

        # Data
        data = self._package_training_data(method, x, theta0, theta1, y, r_xz, t_xz0, t_xz1)

        # Create model
        if self.model is None:
            logger.info("Creating model", method)
            self._create_model()

        # Losses
        loss_functions, loss_labels, loss_weights = get_loss(method + "2", alpha)

        # Optimizer
        opt, opt_kwargs = get_optimizer(optimizer, nesterov_momentum)

        # Train model
        logger.info("Training model")
        trainer = DoubleParameterizedRatioTrainer(self.model)
        result = trainer.train(
            data=data,
            loss_functions=loss_functions,
            loss_weights=loss_weights,
            loss_labels=loss_labels,
            epochs=n_epochs,
            batch_size=batch_size,
            optimizer=opt,
            optimizer_kwargs=opt_kwargs,
            initial_lr=initial_lr,
            final_lr=final_lr,
            validation_split=validation_split,
            early_stopping=early_stopping,
            verbose=verbose,
        )
        return result

    def evaluate_log_likelihood_ratio(self, x, theta0, theta1, test_all_combinations=True, evaluate_score=False):
        """
        Evaluates the log likelihood ratio as a function of the observation x, the numerator hypothesis theta0, and
        the denominator hypothesis theta1.

        Parameters
        ----------
        x : str or ndarray
            Observations or filename of a pickled numpy array.

        theta0 : ndarray or str
            Numerator parameter points or filename of a pickled numpy array.

        theta1 : ndarray or str
            Denominator parameter points or filename of a pickled numpy array.

        test_all_combinations : bool, optional
            If False, the number of samples in the observable and theta
            files has to match, and the likelihood ratio is evaluated only for the combinations
            `r(x_i | theta0_i, theta1_i)`. If True, `r(x_i | theta0_j, theta1_j)` for all pairwise combinations `i, j`
            are evaluated. Default value: True.

        evaluate_score : bool, optional
            Sets whether in addition to the likelihood ratio the score is evaluated. Default value: False.

        Returns
        -------
        log_likelihood_ratio : ndarray
            The estimated log likelihood ratio. If test_all_combinations is True, the result has shape
            `(n_thetas, n_x)`. Otherwise, it has shape `(n_samples,)`.

        score0 : ndarray or None
            None if evaluate_score is False. Otherwise the derived estimated score at `theta0`. If test_all_combinations
            is True, the result has shape `(n_thetas, n_x, n_parameters)`. Otherwise, it has shape
            `(n_samples, n_parameters)`.

        score1 : ndarray or None
            None if evaluate_score is False. Otherwise the derived estimated score at `theta1`. If test_all_combinations
            is True, the result has shape `(n_thetas, n_x, n_parameters)`. Otherwise, it has shape
            `(n_samples, n_parameters)`.

        """
        if self.model is None:
            raise ValueError("No model -- train or load model before evaluating it!")

        # Load training data
        logger.info("Loading evaluation data")
        x = load_and_check(x)
        theta0 = load_and_check(theta0)
        theta1 = load_and_check(theta1)

        # Scale observables
        x = self._transform_inputs(x)

        # Restrict features
        if self.features is not None:
            x = x[:, self.features]

        # Balance thetas
        if len(theta1) > len(theta0):
            theta0 = [theta0[i % len(theta0)] for i in range(len(theta1))]
        elif len(theta1) < len(theta0):
            theta1 = [theta1[i % len(theta1)] for i in range(len(theta0))]

        all_log_r_hat = []
        all_t_hat0 = []
        all_t_hat1 = []

        if test_all_combinations:
            logger.info("Starting ratio evaluation for %s x-theta combinations", len(theta0) * len(x))

            for i, (this_theta0, this_theta1) in enumerate(zip(theta0, theta1)):
                logger.debug(
                    "Starting ratio evaluation for thetas %s / %s: %s vs %s",
                    i + 1,
                    len(theta0),
                    this_theta0,
                    this_theta1,
                )
                _, log_r_hat, t_hat0, t_hat1 = evaluate_ratio_model(
                    model=self.model,
                    method_type="double_parameterized_ratio",
                    theta0s=[this_theta0],
                    theta1s=[this_theta1],
                    xs=x,
                    evaluate_score=evaluate_score,
                )

                all_log_r_hat.append(log_r_hat)
                all_t_hat0.append(t_hat0)
                all_t_hat1.append(t_hat1)

            all_log_r_hat = np.array(all_log_r_hat)
            all_t_hat0 = np.array(all_t_hat0)
            all_t_hat1 = np.array(all_t_hat1)

        else:
            logger.info("Starting ratio evaluation")
            _, all_log_r_hat, all_t_hat0, all_t_hat1 = evaluate_ratio_model(
                model=self.model,
                method_type="double_parameterized_ratio",
                theta0s=theta0,
                theta1s=theta1,
                xs=x,
                evaluate_score=evaluate_score,
            )

        logger.info("Evaluation done")
        return all_log_r_hat, all_t_hat0, all_t_hat1

    def evaluate_log_likelihood(self, *args, **kwargs):
        raise TheresAGoodReasonThisDoesntWork(
            "This estimator can only estimate likelihood ratios, not the likelihood " "itself!"
        )

    def evaluate_score(self, *args, **kwargs):
        raise NotImplementedError("Please use evaluate_log_likelihood_ratio(evaluate_score=True).")

    def calculate_fisher_information(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use evaluate_log_likelihood_ratio(evaluate_score=True) and calculate the "
            "Fisher information manually."
        )

    def evaluate(self, *args, **kwargs):
        return self.evaluate_log_likelihood_ratio(*args, **kwargs)

    def _create_model(self):
        self.model = DenseDoublyParameterizedRatioModel(
            n_observables=self.n_observables,
            n_parameters=self.n_parameters,
            n_hidden=self.n_hidden,
            activation=self.activation,
        )

    @staticmethod
    def _check_required_data(method, r_xz, t_xz0, t_xz1):
        if method in ["cascal", "alices", "rascal"] and (t_xz0 is None or t_xz1 is None):
            raise RuntimeError("Method {} requires joint score information".format(method))
        if method in ["rolr", "alice", "alices", "rascal"] and r_xz is None:
            raise RuntimeError("Method {} requires joint likelihood ratio information".format(method))

    @staticmethod
    def _package_training_data(method, x, theta0, theta1, y, r_xz, t_xz0, t_xz1):
        data = OrderedDict()
        data["x"] = x
        data["theta0"] = theta0
        data["theta1"] = theta1
        data["y"] = y
        if method in ["rolr", "alice", "alices", "rascal"]:
            data["r_xz"] = r_xz
        if method in ["cascal", "alices", "rascal"]:
            data["t_xz0"] = t_xz0
            data["t_xz1"] = t_xz1
        return data

    def _wrap_settings(self):
        settings = super(DoubleParameterizedRatioEstimator, self)._wrap_settings()
        settings["estimator_type"] = "double_parameterized_ratio"
        return settings

    def _unwrap_settings(self, settings):
        super(DoubleParameterizedRatioEstimator, self)._unwrap_settings(settings)

        estimator_type = str(settings["estimator_type"])
        if estimator_type != "double_parameterized_ratio":
            raise RuntimeError("Saved model is an incompatible estimator type {}.".format(estimator_type))


class ScoreEstimator(Estimator):
    """ A neural estimator of the score evaluated at a fixed reference hypothesis as a function of the
     observation x.

    Parameters
    ----------
    features : list of int or None, optional
        Indices of observables (features) that are used as input to the neural networks. If None, all observables
        are used. Default value: None.

    n_hidden : tuple of int, optional
        Units in each hidden layer in the neural networks. If method is 'nde' or 'scandal', this refers to the
        setup of each individual MADE layer. Default value: (100,).

    activation : {'tanh', 'sigmoid', 'relu'}, optional
        Activation function. Default value: 'tanh'.

    """

    def __init__(self, features=None, n_hidden=(100,), activation="tanh"):
        super(ScoreEstimator, self).__init__(features, n_hidden, activation)

        self.nuisance_profile_matrix = None
        self.nuisance_project_matrix = None
        self.nuisance_mode_default = "keep"

    def train(
        self,
        method,
        x,
        t_xz,
        optimizer="amsgrad",
        n_epochs=50,
        batch_size=128,
        initial_lr=0.001,
        final_lr=0.0001,
        nesterov_momentum=None,
        validation_split=0.25,
        early_stopping=True,
        scale_inputs=True,
        shuffle_labels=False,
        limit_samplesize=None,
        memmap=False,
        verbose="some",
    ):

        """
        Trains the network.

        Parameters
        ----------
        method : str
            The inference method used for training. Currently values are 'sally' and 'sallino', but at the training
            stage they are identical. So right now it doesn't matter which one you use.

        x : ndarray or str
            Path to an unweighted sample of observations, as saved by the `madminer.sampling.SampleAugmenter` functions.
            Required for all inference methods.

        t_xz : ndarray or str
            Joint scores at the reference hypothesis, or filename of a pickled numpy array.

        optimizer : {"adam", "amsgrad", "sgd"}, optional
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
            sample is used for training and early stopping is deactivated. Default value: 0.25.

        early_stopping : bool, optional
            Activates early stopping based on the validation loss (only if validation_split is not None). Default value:
            True.

        scale_inputs : bool, optional
            Scale the observables to zero mean and unit variance. Default value: True.

        shuffle_labels : bool, optional
            If True, the labels (`y`, `r_xz`, `t_xz`) are shuffled, while the observations (`x`) remain in their
            normal order. This serves as a closure test, in particular as cross-check against overfitting: an estimator
            trained with shuffle_labels=True should predict to likelihood ratios around 1 and scores around 0.

        limit_samplesize : int or None, optional
            If not None, only this number of samples (events) is used to train the estimator. Default value: None.

        memmap : bool, optional.
            If True, training files larger than 1 GB will not be loaded into memory at once. Default value: False.

        verbose : {"all", "many", "some", "few", "none}, optional
            Determines verbosity of training. Default value: "some".

        Returns
        -------
            None

        """

        if method not in ["sally", "sallino"]:
            logger.warning("Method %s not allowed for score estimators. Using 'sally' instead.", method)
            method = "sally"

        logger.info("Starting training")
        logger.info("  Batch size:             %s", batch_size)
        logger.info("  Optimizer:              %s", optimizer)
        logger.info("  Epochs:                 %s", n_epochs)
        logger.info("  Learning rate:          %s initially, decaying to %s", initial_lr, final_lr)
        if optimizer == "sgd":
            logger.info("  Nesterov momentum:      %s", nesterov_momentum)
        logger.info("  Validation split:       %s", validation_split)
        logger.info("  Early stopping:         %s", early_stopping)
        logger.info("  Scale inputs:           %s", scale_inputs)
        logger.info("  Shuffle labels          %s", shuffle_labels)
        if limit_samplesize is None:
            logger.info("  Samples:                all")
        else:
            logger.info("  Samples:                %s", limit_samplesize)

        # Load training data
        logger.info("Loading training data")
        memmap_threshold = 1.0 if memmap else None
        x = load_and_check(x, memmap_files_larger_than_gb=memmap_threshold)
        t_xz = load_and_check(t_xz, memmap_files_larger_than_gb=memmap_threshold)

        # Infer dimensions of problem
        n_samples = x.shape[0]
        n_observables = x.shape[1]
        n_parameters = t_xz.shape[1]
        logger.info("Found %s samples with %s parameters and %s observables", n_samples, n_parameters, n_observables)

        # Limit sample size
        if limit_samplesize is not None and limit_samplesize < n_samples:
            logger.info("Only using %s of %s training samples", limit_samplesize, n_samples)
            x, t_xz = restrict_samplesize(limit_samplesize, x, t_xz)

        # Scale features
        if scale_inputs:
            logger.info("Rescaling inputs")
            self._initialize_input_transform(x)
            x = self._transform_inputs(x)
        else:
            self._initialize_input_transform(x, False)

        # Shuffle labels
        if shuffle_labels:
            logger.info("Shuffling labels")
            t_xz = shuffle(t_xz)

        # Features
        if self.features is not None:
            x = x[:, self.features]
            logger.info("Only using %s of %s observables", x.shape[1], n_observables)
            n_observables = x.shape[1]

        # Check consistency of input with model
        if self.n_observables is None:
            self.n_observables = n_observables
        if self.n_parameters is None:
            self.n_parameters = n_parameters

        if n_parameters != self.n_parameters:
            raise RuntimeError(
                "Number of parameters does not match model: {} vs {}".format(n_parameters, self.n_parameters)
            )
        if n_observables != self.n_observables:
            raise RuntimeError(
                "Number of observables does not match model: {} vs {}".format(n_observables, self.n_observables)
            )

        # Data
        data = self._package_training_data(x, t_xz)

        # Create model
        if self.model is None:
            logger.info("Creating model")
            self._create_model()

        # Losses
        loss_functions, loss_labels, loss_weights = get_loss(method, None)

        # Optimizer
        opt, opt_kwargs = get_optimizer(optimizer, nesterov_momentum)

        # Train model
        logger.info("Training model")
        trainer = LocalScoreTrainer(self.model)
        result = trainer.train(
            data=data,
            loss_functions=loss_functions,
            loss_weights=loss_weights,
            loss_labels=loss_labels,
            epochs=n_epochs,
            batch_size=batch_size,
            optimizer=opt,
            optimizer_kwargs=opt_kwargs,
            initial_lr=initial_lr,
            final_lr=final_lr,
            validation_split=validation_split,
            early_stopping=early_stopping,
            verbose=verbose,
        )
        return result

    def set_nuisance(self, fisher_information, parameters_of_interest):
        """
        Prepares the calculation of profiled scores, see https://arxiv.org/pdf/1903.01473.pdf.

        Parameters
        ----------
        fisher_information : ndarray
            Fisher informatioin with shape `(n_parameters, n_parameters)`.

        parameters_of_interest : list of int
            List of int, with 0 <= remaining_compoinents[i] < n_parameters. Denotes which parameters are kept in the
            profiling, and their new order.

        Returns
        -------
            None

        """
        if fisher_information.shape != (self.n_parameters, self.n_parameters):
            raise ValueError(
                "Fisher information has wrong shape {}, expected {}".format(
                    fisher_information.shape, (self.n_parameters, self.n_parameters)
                )
            )

        n_parameters_of_interest = len(parameters_of_interest)

        # Separate Fisher information parts
        nuisance_parameters, information_phys, information_mix, information_nuisance = separate_information_blocks(
            fisher_information, parameters_of_interest
        )

        # Calculate projection matrix
        self.nuisance_project_matrix = np.zeros((n_parameters_of_interest, self.n_parameters))  # (n_phys, n_all)
        for theta_new, theta_old in enumerate(parameters_of_interest):
            self.nuisance_project_matrix[theta_new, theta_old] = 1.0

        logger.debug("Nuisance projection matrix:/n%s", self.nuisance_project_matrix)

        # Calculate profiling matrix
        inverse_information_nuisance = np.linalg.inv(information_nuisance)  # (n_nuisance, n_nuisance)
        profiling_matrix = -information_mix.T.dot(inverse_information_nuisance)  # (n_phys, n_nuisance)

        self.nuisance_profile_matrix = np.copy(self.nuisance_project_matrix)  # (n_phys, n_all)
        for theta_new, theta_old in enumerate(parameters_of_interest):
            for nuis_new, nuis_old in enumerate(nuisance_parameters):
                self.nuisance_profile_matrix[theta_new, nuis_old] += profiling_matrix[theta_new, nuis_new]

        logger.debug("Nuisance profiling matrix:/n%s", self.nuisance_project_matrix)

    def evaluate_score(self, x, nuisance_mode="auto"):
        """
        Evaluates the score.

        Parameters
        ----------
        x : str or ndarray
            Observations, or filename of a pickled numpy array.

        nuisance_mode : {"auto", "keep", "profile", "project"}
            Decides how nuisance parameters are treated. If nuisance_mode is "auto", the returned score is the (n+k)-
            dimensional score in the space of n parameters of interest and k nuisance parameters if `set_profiling`
            has not been called, and the n-dimensional profiled score in the space of the parameters of interest
            if it has been called. For "keep", the returned score is always (n+k)-dimensional. For "profile", it is
            the n-dimensional profiled score. For "project", it is the n-dimensional projected score, i.e. ignoring
            the nuisance parameters.

        Returns
        -------
        score : ndarray
            Estimated score with shape `(n_observations, n_parameters)`.
        """

        if self.model is None:
            raise ValueError("No model -- train or load model before evaluating it!")

        if nuisance_mode == "auto":
            logger.debug("Using nuisance mode %s", self.nuisance_mode_default)
            nuisance_mode = self.nuisance_mode_default

        # Load training data
        if isinstance(x, str):
            logger.info("Loading evaluation data")
        x = load_and_check(x)

        # Scale observables
        x = self._transform_inputs(x)

        # Restrict featuers
        if self.features is not None:
            x = x[:, self.features]

        # Evaluation
        logger.info("Starting score evaluation")
        t_hat = evaluate_local_score_model(model=self.model, xs=x)

        # Treatment of nuisance paramters
        if nuisance_mode == "keep":
            logging.info("Keeping nuisance parameter score")

        elif nuisance_mode == "project":
            if self.nuisance_project_matrix is None:
                raise ValueError(
                    "evaluate_score() was called with nuisance_mode = project, but nuisance parameters "
                    "have not been set up yet. Please call set_nuisance() first!"
                )
            logging.info("Projecting nuisance parameter score")
            t_hat = np.einsum("ij,xj->xi", self.nuisance_project_matrix, t_hat)

        elif nuisance_mode == "profile":
            if self.nuisance_profile_matrix is None:
                raise ValueError(
                    "evaluate_score() was called with nuisance_mode = profile, but nuisance parameters "
                    "have not been set up yet. Please call set_nuisance() first!"
                )
            logging.info("Profiling nuisance parameter score")
            t_hat = np.einsum("ij,xj->xi", self.nuisance_profile_matrix, t_hat)

        else:
            raise ValueError("Unknown nuisance_mode {}".format(nuisance_mode))

        return t_hat

    def evaluate_log_likelihood(self, *args, **kwargs):
        raise TheresAGoodReasonThisDoesntWork("This estimator can only estimate the score, not the likelihood!")

    def evaluate_log_likelihood_ratio(self, *args, **kwargs):
        raise TheresAGoodReasonThisDoesntWork("This estimator can only estimate the score, not the likelihood ratio!")

    def evaluate(self, *args, **kwargs):
        return self.evaluate_score(*args, **kwargs)

    def calculate_fisher_information(self, x, weights=None, n_events=1, sum_events=True):
        """
        Calculates the expected Fisher information matrix based on the kinematic information in a given number of
        events.

        Parameters
        ----------
        x : str or ndarray
            Sample of observations, or path to numpy file with observations. Note that this sample has to be sampled
            from the reference parameter where the score is estimated with the SALLY / SALLINO estimator.

        weights : None or ndarray, optional
            Weights for the observations. If None, all events are taken to have equal weight. Default value: None.

        n_events : float, optional
            Expected number of events for which the kinematic Fisher information should be calculated. Default value: 1.

        sum_events : bool, optional
            If True, the expected Fisher information summed over the events x is calculated. If False, the per-event
            Fisher information for each event is returned. Default value: True.

        Returns
        -------
        fisher_information : ndarray
            Expected kinematic Fisher information matrix with shape `(n_events, n_parameters, n_parameters)` if
            sum_events is False or `(n_parameters, n_parameters)` if sum_events is True.

        """
        if self.model is None:
            raise ValueError("No model -- train or load model before evaluating it!")

        # Load training data
        logger.info("Loading evaluation data")
        x = load_and_check(x)
        n_samples = x.shape[0]

        # Scale observables
        x = self._transform_inputs(x)

        # Restrict featuers
        if self.features is not None:
            x = x[:, self.features]

        # Estimate scores
        t_hats = evaluate_local_score_model(model=self.model, xs=x)

        # Weights
        if weights is None:
            weights = np.ones(n_samples)
        weights /= np.sum(weights)

        # Calculate Fisher information
        logger.info("Calculating Fisher information")
        if sum_events:
            fisher_information = float(n_events) * np.einsum("n,ni,nj->ij", weights, t_hats, t_hats)
        else:
            fisher_information = float(n_events) * np.einsum("n,ni,nj->nij", weights, t_hats, t_hats)

        # Calculate expected score
        expected_score = np.mean(t_hats, axis=0)
        logger.debug("Expected per-event score (should be close to zero): %s", expected_score)

        return fisher_information

    def save(self, filename, save_model=False):
        super(ScoreEstimator, self).save(filename, save_model)

        # Also save Fisher information information for profiling / projections
        if self.nuisance_profile_matrix is not None and self.nuisance_project_matrix is not None:
            logger.debug(
                "Saving nuisance profiling / projection information to %s_nuisance_profile_matrix.npy and "
                "%s_nuisance_project_matrix.npy",
                filename,
                filename,
            )
            np.save(filename + "_nuisance_profile_matrix.npy", self.nuisance_profile_matrix)
            np.save(filename + "_nuisance_project_matrix.npy", self.nuisance_project_matrix)

    def load(self, filename):
        super(ScoreEstimator, self).load(filename)

        # Load scaling
        try:
            self.nuisance_profile_matrix = np.load(filename + "_nuisance_profile_matrix.npy")
            self.nuisance_project_matrix = np.load(filename + "_nuisance_project_matrix.npy")
            logger.debug(
                "  Found nuisance profiling / projection matrices:\nProfiling:\n%s\nProjection:\n%s",
                self.nuisance_profile_matrix,
                self.nuisance_project_matrix,
            )
        except:
            logger.debug("Did not find nuisance profiling / projection setup in %s", filename)
            self.nuisance_profile_matrix = None
            self.nuisance_project_matrix = None

    def _create_model(self):
        self.model = DenseLocalScoreModel(
            n_observables=self.n_observables,
            n_parameters=self.n_parameters,
            n_hidden=self.n_hidden,
            activation=self.activation,
        )

    @staticmethod
    def _package_training_data(x, t_xz):
        data = OrderedDict()
        data["x"] = x
        data["t_xz"] = t_xz
        return data

    def _wrap_settings(self):
        settings = super(ScoreEstimator, self)._wrap_settings()
        settings["estimator_type"] = "score"
        settings["estimator_type"] = "score"
        settings["nuisance_mode_default"] = self.nuisance_mode_default
        return settings

    def _unwrap_settings(self, settings):
        super(ScoreEstimator, self)._unwrap_settings(settings)

        estimator_type = str(settings["estimator_type"])
        if estimator_type != "score":
            raise RuntimeError("Saved model is an incompatible estimator type {}.".format(estimator_type))

        try:
            self.nuisance_mode_default = str(settings["nuisance_mode_default"])
        except KeyError:
            self.nuisance_mode_default = "keep"
            logger.warning("Did not find entry nuisance_mode_default in saved model, using default 'keep'.")


class LikelihoodEstimator(Estimator):
    """ A neural estimator of the density or likelihood evaluated at a reference hypothesis as a function
     of the observation x.

    Parameters
    ----------
    features : list of int or None, optional
        Indices of observables (features) that are used as input to the neural networks. If None, all observables
        are used. Default value: None.

    n_components : int, optional
        The number of Gaussian base components in a MADE MoG. If 1, a plain MADE is used.
        Default value: 1.

    n_mades : int, optional
        The number of MADE layers. Default value: 3.


    n_hidden : tuple of int, optional
        Units in each hidden layer in the neural networks. If method is 'nde' or 'scandal', this refers to the
        setup of each individual MADE layer. Default value: (100,).

    activation : {'tanh', 'sigmoid', 'relu'}, optional
        Activation function. Default value: 'tanh'.

    batch_norm : None or floar, optional
        If not None, batch normalization is used, where this value sets the alpha parameter in the calculation
        of the running average of the mean and variance. Default value: None.


    """

    def __init__(self, features=None, n_components=1, n_mades=5, n_hidden=(100,), activation="tanh", batch_norm=None):
        super(LikelihoodEstimator, self).__init__(features, n_hidden, activation)

        self.n_components = n_components
        self.n_mades = n_mades
        self.batch_norm = batch_norm

    def train(
        self,
        method,
        x,
        theta,
        t_xz=None,
        alpha=1.0,
        optimizer="amsgrad",
        n_epochs=50,
        batch_size=128,
        initial_lr=0.001,
        final_lr=0.0001,
        nesterov_momentum=None,
        validation_split=0.25,
        early_stopping=True,
        scale_inputs=True,
        shuffle_labels=False,
        limit_samplesize=None,
        memmap=False,
        verbose="some",
    ):

        """
        Trains the network.

        Parameters
        ----------
        method : str
            The inference method used for training. Allowed values are 'nde' and 'scandal'.

        x : ndarray or str
            Path to an unweighted sample of observations, as saved by the `madminer.sampling.SampleAugmenter` functions.
            Required for all inference methods.

        theta : ndarray or str
            Numerator parameter point, or filename of a pickled numpy array.

        t_xz : ndarray or str or None, optional
            Joint scores at theta, or filename of a pickled numpy array. Default value: None.

        alpha : float, optional
            Hyperparameter weighting the score error in the loss function of the 'alices', 'rascal', and 'cascal'
            methods. Default value: 1.

        optimizer : {"adam", "amsgrad", "sgd"}, optional
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
            sample is used for training and early stopping is deactivated. Default value: 0.25.

        early_stopping : bool, optional
            Activates early stopping based on the validation loss (only if validation_split is not None). Default value:
            True.

        scale_inputs : bool, optional
            Scale the observables to zero mean and unit variance. Default value: True.

        shuffle_labels : bool, optional
            If True, the labels (`y`, `r_xz`, `t_xz`) are shuffled, while the observations (`x`) remain in their
            normal order. This serves as a closure test, in particular as cross-check against overfitting: an estimator
            trained with shuffle_labels=True should predict to likelihood ratios around 1 and scores around 0.

        limit_samplesize : int or None, optional
            If not None, only this number of samples (events) is used to train the estimator. Default value: None.

        memmap : bool, optional.
            If True, training files larger than 1 GB will not be loaded into memory at once. Default value: False.

        verbose : {"all", "many", "some", "few", "none}, optional
            Determines verbosity of training. Default value: "some".

        Returns
        -------
            None

        """

        logger.info("Starting training")
        logger.info("  Method:                 %s", method)
        if method == "scandal":
            logger.info("  alpha:                  %s", alpha)
        logger.info("  Batch size:             %s", batch_size)
        logger.info("  Optimizer:              %s", optimizer)
        logger.info("  Epochs:                 %s", n_epochs)
        logger.info("  Learning rate:          %s initially, decaying to %s", initial_lr, final_lr)
        if optimizer == "sgd":
            logger.info("  Nesterov momentum:      %s", nesterov_momentum)
        logger.info("  Validation split:       %s", validation_split)
        logger.info("  Early stopping:         %s", early_stopping)
        logger.info("  Scale inputs:           %s", scale_inputs)
        logger.info("  Shuffle labels          %s", shuffle_labels)
        if limit_samplesize is None:
            logger.info("  Samples:                all")
        else:
            logger.info("  Samples:                %s", limit_samplesize)

        # Load training data
        logger.info("Loading training data")
        memmap_threshold = 1.0 if memmap else None
        theta = load_and_check(theta, memmap_files_larger_than_gb=memmap_threshold)
        x = load_and_check(x, memmap_files_larger_than_gb=memmap_threshold)
        t_xz = load_and_check(t_xz, memmap_files_larger_than_gb=memmap_threshold)

        self._check_required_data(method, t_xz)

        # Infer dimensions of problem
        n_samples = x.shape[0]
        n_observables = x.shape[1]
        n_parameters = theta.shape[1]
        logger.info("Found %s samples with %s parameters and %s observables", n_samples, n_parameters, n_observables)

        # Limit sample size
        if limit_samplesize is not None and limit_samplesize < n_samples:
            logger.info("Only using %s of %s training samples", limit_samplesize, n_samples)
            x, theta, t_xz = restrict_samplesize(limit_samplesize, x, theta, t_xz)

        # Scale features
        if scale_inputs:
            logger.info("Rescaling inputs")
            self._initialize_input_transform(x)
            x = self._transform_inputs(x)
        else:
            self._initialize_input_transform(x, False)

        # Shuffle labels
        if shuffle_labels:
            logger.info("Shuffling labels")
            t_xz = shuffle(t_xz)

        # Features
        if self.features is not None:
            x = x[:, self.features]
            logger.info("Only using %s of %s observables", x.shape[1], n_observables)
            n_observables = x.shape[1]

        # Check consistency of input with model
        if self.n_observables is None:
            self.n_observables = n_observables
        if self.n_parameters is None:
            self.n_parameters = n_parameters

        if n_parameters != self.n_parameters:
            raise RuntimeError(
                "Number of parameters does not match model: {} vs {}".format(n_parameters, self.n_parameters)
            )
        if n_observables != self.n_observables:
            raise RuntimeError(
                "Number of observables does not match model: {} vs {}".format(n_observables, self.n_observables)
            )

        # Data
        data = self._package_training_data(method, x, theta, t_xz)

        # Create model
        if self.model is None:
            self._create_model()

        # Losses
        loss_functions, loss_labels, loss_weights = get_loss(method, alpha)

        # Optimizer
        opt, opt_kwargs = get_optimizer(optimizer, nesterov_momentum)

        # Train model
        logger.info("Training model")
        trainer = FlowTrainer(self.model)
        result = trainer.train(
            data=data,
            loss_functions=loss_functions,
            loss_weights=loss_weights,
            loss_labels=loss_labels,
            epochs=n_epochs,
            batch_size=batch_size,
            optimizer=opt,
            optimizer_kwargs=opt_kwargs,
            initial_lr=initial_lr,
            final_lr=final_lr,
            validation_split=validation_split,
            early_stopping=early_stopping,
            verbose=verbose,
        )
        return result

    def evaluate_log_likelihood(self, x, theta, test_all_combinations=True, evaluate_score=False):

        """
        Evaluates the log likelihood as a function of the observation x and the parameter point theta.

        Parameters
        ----------
        x : ndarray or str
            Sample of observations, or path to numpy file with observations.

        theta : ndarray or str
            Parameter points, or path to numpy file with parameter points.

        test_all_combinations : bool, optional
            If method is not 'sally' and not 'sallino': If False, the number of samples in the observable and theta
            files has to match, and the likelihood ratio is evaluated only for the combinations
            `r(x_i | theta0_i, theta1_i)`. If True, `r(x_i | theta0_j, theta1_j)` for all pairwise combinations `i, j`
            are evaluated. Default value: True.

        evaluate_score : bool, optional
            If method is not 'sally' and not 'sallino', this sets whether in addition to the likelihood ratio the score
            is evaluated. Default value: False.

        Returns
        -------

        log_likelihood : ndarray
            The estimated log likelihood. If test_all_combinations is True, the result has shape `(n_thetas, n_x)`.
            Otherwise, it has shape `(n_samples,)`.

        score : ndarray or None
            None if
            evaluate_score is False. Otherwise the derived estimated score at `theta`. If test_all_combinations is
            True, the result has shape `(n_thetas, n_x, n_parameters)`. Otherwise, it has shape
            `(n_samples, n_parameters)`.

        """

        if self.model is None:
            raise ValueError("No model -- train or load model before evaluating it!")

        # Load training data
        if isinstance(x, str):
            logger.info("Loading evaluation data")
        theta = load_and_check(theta)
        x = load_and_check(x)

        # Scale observables
        x = self._transform_inputs(x)

        # Restrict featuers
        if self.features is not None:
            x = x[:, self.features]

        # Evaluation for all other methods
        all_log_p_hat = []
        all_t_hat = []

        if test_all_combinations:
            logger.info("Starting ratio evaluation for %s x-theta combinations", len(theta) * len(x))

            for i, this_theta in enumerate(theta):
                logger.debug("Starting log likelihood evaluation for thetas %s / %s: %s", i + 1, len(theta), this_theta)

                log_p_hat, t_hat = evaluate_flow_model(
                    model=self.model, thetas=[this_theta], xs=x, evaluate_score=evaluate_score
                )

                all_log_p_hat.append(log_p_hat)
                all_t_hat.append(t_hat)

            all_log_p_hat = np.array(all_log_p_hat)
            all_t_hat = np.array(all_t_hat)

        else:
            logger.info("Starting log likelihood evaluation")

            all_log_p_hat, all_t_hat = evaluate_flow_model(
                model=self.model, thetas=theta, xs=x, evaluate_score=evaluate_score
            )

        logger.info("Evaluation done")
        return all_log_p_hat, all_t_hat

    def evaluate_log_likelihood_ratio(self, x, theta0, theta1, test_all_combinations, evaluate_score=False):

        """
        Evaluates the log likelihood ratio as a function of the observation x, the numerator parameter point theta0,
        and the denominator parameter point theta1.

        Parameters
        ----------
        x : ndarray or str
            Sample of observations, or path to numpy file with observations.

        theta0 : ndarray or str
            Numerator parameters, or path to numpy file.

        theta1 : ndarray or str
            Denominator parameters, or path to numpy file.

        test_all_combinations : bool, optional
            If method is not 'sally' and not 'sallino': If False, the number of samples in the observable and theta
            files has to match, and the likelihood ratio is evaluated only for the combinations
            `r(x_i | theta0_i, theta1_i)`. If True, `r(x_i | theta0_j, theta1_j)` for all pairwise combinations `i, j`
            are evaluated. Default value: True.

        evaluate_score : bool, optional
            If method is not 'sally' and not 'sallino', this sets whether in addition to the likelihood ratio the score
            is evaluated. Default value: False.

        Returns
        -------

        log_likelihood : ndarray
            The estimated log likelihood. If test_all_combinations is True, the result has shape `(n_thetas, n_x)`.
            Otherwise, it has shape `(n_samples,)`.

        score : ndarray or None
            None if
            evaluate_score is False. Otherwise the derived estimated score at `theta`. If test_all_combinations is
            True, the result has shape `(n_thetas, n_x, n_parameters)`. Otherwise, it has shape
            `(n_samples, n_parameters)`.

        """

        if self.model is None:
            raise ValueError("No model -- train or load model before evaluating it!")

        # Load training data
        logger.info("Loading evaluation data")
        x = load_and_check(x)
        theta0 = load_and_check(theta0)
        theta1 = load_and_check(theta1)

        # Scale observables
        x = self._transform_inputs(x)

        # Restrict features
        if self.features is not None:
            x = x[:, self.features]

        # Balance thetas
        if len(theta1) > len(theta0):
            theta0 = [theta0[i % len(theta0)] for i in range(len(theta1))]
        elif len(theta1) < len(theta0):
            theta1 = [theta1[i % len(theta1)] for i in range(len(theta0))]

        log_p_hat0, t_hat0 = self.evaluate_log_likelihood(
            x, theta0, test_all_combinations=test_all_combinations, evaluate_score=evaluate_score
        )
        log_p_hat1, t_hat1 = self.evaluate_log_likelihood(
            x, theta1, test_all_combinations=test_all_combinations, evaluate_score=evaluate_score
        )
        log_r_hat = log_p_hat0 - log_p_hat1

        return log_r_hat, t_hat0, t_hat1

    def evaluate_score(self, *args, **kwargs):
        raise NotImplementedError("Please use evaluate_log_likelihood(evaluate_score=True).")

    def calculate_fisher_information(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use evaluate_log_likelihood_ratio(evaluate_score=True) and calculate the "
            "Fisher information manually."
        )

    def evaluate(self, *args, **kwargs):
        return self.evaluate_log_likelihood(*args, **kwargs)

    def _create_model(self):
        if self.n_components > 1:
            self.model = ConditionalMixtureMaskedAutoregressiveFlow(
                n_conditionals=self.n_parameters,
                n_inputs=self.n_observables,
                n_components=self.n_components,
                n_hiddens=self.n_hidden,
                n_mades=self.n_mades,
                activation=self.activation,
                batch_norm=self.batch_norm is not None,
                alpha=self.batch_norm,
            )
        else:
            self.model = ConditionalMaskedAutoregressiveFlow(
                n_conditionals=self.n_parameters,
                n_inputs=self.n_observables,
                n_hiddens=self.n_hidden,
                n_mades=self.n_mades,
                activation=self.activation,
                batch_norm=self.batch_norm is not None,
                alpha=self.batch_norm,
            )

    @staticmethod
    def _check_required_data(method, t_xz):
        if method == ["scandal"] and t_xz is None:
            raise RuntimeError("Method {} requires joint score information".format(method))

    @staticmethod
    def _package_training_data(method, x, theta, t_xz):
        data = OrderedDict()
        data["x"] = x
        data["theta"] = theta
        if method in ["scandal"]:
            data["t_xz"] = t_xz
        return data

    def _wrap_settings(self):
        settings = super(LikelihoodEstimator, self)._wrap_settings()
        settings["estimator_type"] = "likelihood"
        settings["n_components"] = self.n_components
        settings["batch_norm"] = self.batch_norm
        settings["n_mades"] = self.n_mades
        return settings

    def _unwrap_settings(self, settings):
        super(LikelihoodEstimator, self)._unwrap_settings(settings)

        estimator_type = str(settings["estimator_type"])
        if estimator_type != "likelihood":
            raise RuntimeError("Saved model is an incompatible estimator type {}.".format(estimator_type))

        self.n_components = int(settings["n_components"])
        self.n_mades = int(settings["n_mades"])
        self.batch_norm = settings["batch_norm"]
        if self.batch_norm == "None":
            self.batch_norm = None
        if self.batch_norm is not None:
            self.batch_norm = float(self.batch_norm)


class Ensemble:
    """
    Ensemble methods for likelihood, likelihood ratio, and score estimation.

    Generally, Ensemble instances can be used very similarly to Estimator instances:

    * The initialization of Ensemble takes a list of (trained or untrained) Estimator instances.
    * The methods `Ensemble.train_one()` and `Ensemble.train_all()` train the estimators (this can also be
      done outside of Ensemble).
    * `Ensemble.calculate_expectation()` can be used to calculate the expectation of the estimation likelihood
      ratio or the expected estimated score over a validation sample. Ideally (and assuming the correct sampling),
      these expectation values should be close to zero. Deviations from zero therefore point out that the estimator
      is probably inaccurate.
    * `Ensemble.evaluate_log_likelihood()`, `Ensemble.evaluate_log_likelihood_ratio()`, `Ensemble.evaluate_score()`,
      and `Ensemble.calculate_fisher_information()` can then be used to calculate
      ensemble predictions.
    * `Ensemble.save()` and `Ensemble.load()` can store all estimators in one folder.

    The individual estimators in the ensemble can be trained with different methods, but they have to be of the same
    type: either all estimators are ParameterizedRatioEstimator instances, or all estimators are
    DoubleParameterizedRatioEstimator instances, or all estimators are ScoreEstimator instances, or all estimators are
    LikelihoodEstimator instances..

    Parameters
    ----------
    estimators : None or list of Estimator, optional
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
    estimators : list of Estimator
        The estimators in the form of MLForge instances.
    """

    def __init__(self, estimators=None):
        self.n_parameters = None
        self.n_observables = None
        self.estimator_type = None

        # Initialize estimators
        if estimators is None:
            self.estimators = []
        else:
            self.estimators = []
            for estimator in estimators:
                if isinstance(estimator, Estimator):
                    self.estimators.append(estimator)
                else:
                    raise ValueError("Entry {} in estimators is neither str nor Estimator instance")

        self.n_estimators = len(self.estimators)
        self._check_consistency()

    def add_estimator(self, estimator):
        """
        Adds an estimator to the ensemble.

        Parameters
        ----------
        estimator : Estimator
            The estimator.

        Returns
        -------
            None

        """
        if not isinstance(estimator, Estimator):
            raise ValueError("Entry {} in estimators is neither str nor Estimator instance")

        self.estimators.append(estimator)
        self.n_estimators = len(self.estimators)
        self._check_consistency()

    def train_one(self, i, **kwargs):
        """
        Trains an individual estimator. See `Estimator.train()`.

        Parameters
        ----------
        i : int
            The index `0 <= i < n_estimators` of the estimator to be trained.

        kwargs : dict
            Parameters for `Estimator.train()`.

        Returns
        -------
            None

        """

        self.estimators[i].train(**kwargs)

    def train_all(self, **kwargs):
        """
        Trains all estimators. See `Estimator.train()`.

        Parameters
        ----------
        kwargs : dict
            Parameters for `Estimator.train()`. If a value in this dict is a list, it has to have length `n_estimators`
            and contain one value of this parameter for each of the estimators. Otherwise the value is used as parameter
            for the training of all the estimators.

        Returns
        -------
            None

        """
        logger.info("Training %s estimators in ensemble", self.n_estimators)

        for key, value in six.iteritems(kwargs):
            if not isinstance(value, list):
                kwargs[key] = [value for _ in range(self.n_estimators)]

            assert len(kwargs[key]) == self.n_estimators, "Keyword {} has wrong length {}".format(key, len(value))

        for i, estimator in enumerate(self.estimators):
            kwargs_this_estimator = {}
            for key, value in six.iteritems(kwargs):
                kwargs_this_estimator[key] = value[i]

            logger.info("Training estimator %s / %s in ensemble", i + 1, self.n_estimators)
            estimator.train(**kwargs_this_estimator)

    def evaluate_log_likelihood(self, estimator_weights=None, calculate_covariance=False, **kwargs):
        """
        Estimates the log likelihood from each estimator and returns the ensemble mean (and, if calculate_covariance is
        True, the covariance between them).

        Parameters
        ----------
        estimator_weights : ndarray or None, optional
            Weights for each estimator in the ensemble. If None, all estimators have an equal vote. Default value: None.

        calculate_covariance : bool, optional
            If True, the covariance between the different estimators is calculated. Default value: False.

        kwargs
            Arguments for the evaluation. See the documentation of the relevant Estimator class.

        Returns
        -------
        log_likelihood : ndarray
            Mean prediction for the log likelihood.

        covariance : ndarray or None
            If calculate_covariance is True, the covariance matrix between the estimators. Otherwise None.

        """

        logger.info("Evaluating %s estimators in ensemble", self.n_estimators)

        # Calculate weights of each estimator in vote
        if estimator_weights is None:
            estimator_weights = np.ones(self.n_estimators)
        assert len(estimator_weights) == self.n_estimators
        estimator_weights /= np.sum(estimator_weights)
        logger.debug("Estimator weights: %s", estimator_weights)

        # Calculate estimator predictions
        predictions = []
        for i, estimator in enumerate(self.estimators):
            logger.info("Starting evaluation for estimator %s / %s in ensemble", i + 1, self.n_estimators)
            predictions.append(estimator.evaluate_log_likelihood(**kwargs)[0])
        predictions = np.array(predictions)

        # Calculate weighted means and covariance matrices
        mean = np.average(predictions, axis=0, weights=estimator_weights)

        if calculate_covariance:
            predictions_flat = predictions.reshape((predictions.shape[0], -1))
            covariance = np.cov(predictions_flat.T, aweights=estimator_weights)
            covariance = covariance.reshape(list(predictions.shape) + list(predictions.shape))
        else:
            covariance = None

        return mean, covariance

    def evaluate_log_likelihood_ratio(self, estimator_weights=None, calculate_covariance=False, **kwargs):
        """
        Estimates the log likelihood ratio from each estimator and returns the ensemble mean (and, if
        calculate_covariance is True, the covariance between them).

        Parameters
        ----------
        estimator_weights : ndarray or None, optional
            Weights for each estimator in the ensemble. If None, all estimators have an equal vote. Default value: None.

        calculate_covariance : bool, optional
            If True, the covariance between the different estimators is calculated. Default value: False.

        kwargs
            Arguments for the evaluation. See the documentation of the relevant Estimator class.

        Returns
        -------
        log_likelihood_ratio : ndarray
            Mean prediction for the log likelihood ratio.

        covariance : ndarray or None
            If calculate_covariance is True, the covariance matrix between the estimators. Otherwise None.

        """

        logger.info("Evaluating %s estimators in ensemble", self.n_estimators)

        # Calculate weights of each estimator in vote
        if estimator_weights is None:
            estimator_weights = np.ones(self.n_estimators)
        assert len(estimator_weights) == self.n_estimators
        estimator_weights /= np.sum(estimator_weights)
        logger.debug("Estimator weights: %s", estimator_weights)

        # Calculate estimator predictions
        predictions = []
        for i, estimator in enumerate(self.estimators):
            logger.info("Starting evaluation for estimator %s / %s in ensemble", i + 1, self.n_estimators)
            predictions.append(estimator.evaluate_log_likelihood_ratio(**kwargs)[0])
        predictions = np.array(predictions)

        # Calculate weighted means and covariance matrices
        mean = np.average(predictions, axis=0, weights=estimator_weights)

        if calculate_covariance:
            predictions_flat = predictions.reshape((predictions.shape[0], -1))
            covariance = np.cov(predictions_flat.T, aweights=estimator_weights)
            covariance = covariance.reshape(list(predictions.shape) + list(predictions.shape))
        else:
            covariance = None

        return mean, covariance

    def evaluate_score(self, estimator_weights=None, calculate_covariance=False, **kwargs):
        """
        Estimates the score from each estimator and returns the ensemble mean (and, if
        calculate_covariance is True, the covariance between them).

        Parameters
        ----------
        estimator_weights : ndarray or None, optional
            Weights for each estimator in the ensemble. If None, all estimators have an equal vote. Default value: None.

        calculate_covariance : bool, optional
            If True, the covariance between the different estimators is calculated. Default value: False.

        kwargs
            Arguments for the evaluation. See the documentation of the relevant Estimator class.

        Returns
        -------
        log_likelihood_ratio : ndarray
            Mean prediction for the log likelihood ratio.

        covariance : ndarray or None
            If calculate_covariance is True, the covariance matrix between the estimators. Otherwise None.

        """

        logger.info("Evaluating %s estimators in ensemble", self.n_estimators)

        # Calculate weights of each estimator in vote
        if estimator_weights is None:
            estimator_weights = np.ones(self.n_estimators)
        assert len(estimator_weights) == self.n_estimators
        estimator_weights /= np.sum(estimator_weights)
        logger.debug("Estimator weights: %s", estimator_weights)

        # Calculate estimator predictions
        predictions = []
        for i, estimator in enumerate(self.estimators):
            logger.info("Starting evaluation for estimator %s / %s in ensemble", i + 1, self.n_estimators)
            predictions.append(estimator.evaluate_score(**kwargs))
        predictions = np.array(predictions)

        # Calculate weighted means and covariance matrices
        mean = np.average(predictions, axis=0, weights=estimator_weights)

        if calculate_covariance:
            predictions_flat = predictions.reshape((predictions.shape[0], -1))
            covariance = np.cov(predictions_flat.T, aweights=estimator_weights)
            covariance = covariance.reshape(list(predictions.shape) + list(predictions.shape))
        else:
            covariance = None

        return mean, covariance

    def calculate_fisher_information(
        self,
        x,
        obs_weights=None,
        estimator_weights=None,
        n_events=1,
        mode="score",
        calculate_covariance=True,
        sum_events=True,
    ):
        """
        Calculates expected Fisher information matrices for an ensemble of ScoreEstimator instances.

        There are two ways of calculating the ensemble average. In the default "score" mode, the ensemble average for
        the score is calculated for each event, and the Fisher information is calculated based on these mean scores. In
        the "information" mode, the Fisher information is calculated for each estimator separately and the ensemble
        mean is calculated only for the final Fisher information matrix. The "score" mode is generally assumed to be
        more precise and is the default.

        In the "score" mode, the covariance matrix of the final result is calculated in the following way:
        - For each event `x` and each estimator `a`, the "shifted" predicted score is calculated as
          `t_a'(x) = t(x) + 1/sqrt(n) * (t_a(x) - t(x))`. Here `t(x)` is the mean score (averaged over the ensemble)
          for this event, `t_a(x)` is the prediction of estimator `a` for this event, and `n` is the number of
          estimators. The ensemble variance of these shifted score predictions is equal to the uncertainty on the mean
          of the ensemble of original predictions.
        - For each estimator `a`, the shifted Fisher information matrix `I_a'` is calculated  from the shifted predicted
          scores.
        - The ensemble covariance between all Fisher information matrices `I_a'` is calculated and taken as the
          measure of uncertainty on the Fisher information calculated from the mean scores.

        In the "information" mode, the user has the option to treat all estimators equally ('committee method') or to
        give those with expected score close to zero (as calculated by `calculate_expectation()`) a higher weight. In
        this case, the ensemble mean `I` is calculated as `I  =  sum_i w_i I_i` with weights
        `w_i  =  exp(-vote_expectation_weight |E[t_i]|) / sum_j exp(-vote_expectation_weight |E[t_k]|)`. Here `I_i`
        are the individual estimators and `E[t_i]` is the expectation value calculated by `calculate_expectation()`.

        Parameters
        ----------
        x : str or ndarray
            Sample of observations, or path to numpy file with observations, as saved by the
            `madminer.sampling.SampleAugmenter` functions. Note that this sample has to be sampled from the reference
            parameter where the score is estimated with the SALLY / SALLINO estimator!

        obs_weights : None or ndarray, optional
            Weights for the observations. If None, all events are taken to have equal weight. Default value: None.

        estimator_weights : ndarray or None, optional
            Weights for each estimator in the ensemble. If None, all estimators have an equal vote. Default value: None.

        n_events : float, optional
            Expected number of events for which the kinematic Fisher information should be calculated. Default value: 1.

        mode : {"score", "information"}, optional
            If mode is "information", the Fisher information for each estimator is calculated individually and only then
            are the sample mean and covariance calculated. If mode is "score", the sample mean is
            calculated for the score for each event. Default value: "score".

        calculate_covariance : bool, optional
            If True, the covariance between the different estimators is calculated. Default value: True.

        sum_events : bool, optional
            If True or mode is "information", the expected Fisher information summed over the events x is calculated.
            If False and mode is "score", the per-event Fisher information for each event is returned. Default value:
            True.

        Returns
        -------
        mean_prediction : ndarray
            Expected kinematic Fisher information matrix with shape `(n_events, n_parameters, n_parameters)` if
            sum_events is False and mode is "score", or `(n_parameters, n_parameters)` in any other case.

        covariance : ndarray or None
            The covariance of the estimated Fisher information matrix. This object has four indices, `cov_(ij)(i'j')`,
            ordered as i j i' j'. It has shape `(n_parameters, n_parameters, n_parameters, n_parameters)`.
        """
        logger.debug("Evaluating Fisher information for %s estimators in ensemble", self.n_estimators)

        # Check ensemble
        if self.estimator_type != "score":
            raise NotImplementedError(
                "Fisher information calculation is only implemented for local score estimators "
                "(ScoreEstimator instances)."
            )

        # Check input
        if mode not in ["score", "information"]:
            raise ValueError("Unknown mode {}, has to be 'score' or 'information'!".format(mode))

        # Calculate estimator_weights of each estimator in vote
        if estimator_weights is None:
            estimator_weights = np.ones(self.n_estimators)
        assert len(estimator_weights) == self.n_estimators
        estimator_weights /= np.sum(estimator_weights)
        logger.debug("Estimator weights: %s", estimator_weights)

        covariance = None

        # "information" mode
        if mode == "information":
            # Calculate estimator predictions
            predictions = []
            for i, estimator in enumerate(self.estimators):
                logger.debug("Starting evaluation for estimator %s / %s in ensemble", i + 1, self.n_estimators)

                predictions.append(estimator.calculate_fisher_information(x=x, weights=obs_weights, n_events=n_events))
            predictions = np.array(predictions)

            # Calculate weighted mean and covariance
            information = np.average(predictions, axis=0, weights=estimator_weights)

            predictions_flat = predictions.reshape((predictions.shape[0], -1))

            if calculate_covariance:
                covariance = np.cov(predictions_flat.T, aweights=estimator_weights)
                covariance_shape = (
                    predictions.shape[1],
                    predictions.shape[2],
                    predictions.shape[1],
                    predictions.shape[2],
                )
                covariance = covariance.reshape(covariance_shape)

        # "score" mode:
        else:
            # Load training data
            if isinstance(x, six.string_types):
                x = load_and_check(x)
            n_samples = x.shape[0]

            # Calculate score predictions
            score_predictions = []
            for i, estimator in enumerate(self.estimators):
                logger.debug("Starting evaluation for estimator %s / %s in ensemble", i + 1, self.n_estimators)

                score_predictions.append(estimator.evaluate(x=x))
                logger.debug("Estimator %s predicts t(x) = %s for first event", i + 1, score_predictions[-1][0, :])
            score_predictions = np.array(score_predictions)  # (n_estimators, n_events, n_parameters)

            # Get ensemble mean and ensemble covariance
            score_mean = np.mean(score_predictions, axis=0)  # (n_events, n_parameters)

            # For uncertainty calculation: calculate points betweeen mean and original predictions with same mean and
            # variance / n compared to the original predictions
            score_shifted_predictions = (score_predictions - score_mean[np.newaxis, :, :]) / self.n_estimators ** 0.5
            score_shifted_predictions = score_mean[np.newaxis, :, :] + score_shifted_predictions

            # Event weights
            if obs_weights is None:
                obs_weights = np.ones(n_samples)
            obs_weights /= np.sum(obs_weights)

            # Fisher information prediction (based on mean scores)
            if sum_events:
                information = float(n_events) * np.sum(
                    obs_weights[:, np.newaxis, np.newaxis]
                    * score_mean[:, :, np.newaxis]
                    * score_mean[:, np.newaxis, :],
                    axis=0,
                )
            else:
                information = (
                    float(n_events)
                    * obs_weights[:, np.newaxis, np.newaxis]
                    * score_mean[:, :, np.newaxis]
                    * score_mean[:, np.newaxis, :]
                )

            if calculate_covariance:
                # Fisher information predictions based on shifted scores
                informations_shifted = float(n_events) * np.sum(
                    obs_weights[np.newaxis, :, np.newaxis, np.newaxis]
                    * score_shifted_predictions[:, :, :, np.newaxis]
                    * score_shifted_predictions[:, :, np.newaxis, :],
                    axis=1,
                )  # (n_estimators, n_parameters, n_parameters)

                n_params = score_mean.shape[1]
                informations_shifted = informations_shifted.reshape(-1, n_params ** 2)
                covariance = np.cov(informations_shifted.T)
                covariance = covariance.reshape(n_params, n_params, n_params, n_params)

            # Let's check the expected score
            expected_score = [np.einsum("n,ni->i", obs_weights, score_mean)]
            logger.debug("Expected per-event score (should be close to zero):\n%s", expected_score)

        return information, covariance

    def save(self, folder, save_model=False):
        """
        Saves the estimator ensemble to a folder.

        Parameters
        ----------
        folder : str
            Path to the folder.

        save_model : bool, optional
            If True, the whole model is saved in addition to the state dict. This is not necessary for loading it
            again with Ensemble.load(), but can be useful for debugging, for instance to plot the computational
            graph.

        Returns
        -------
            None

        """

        # Check paths
        create_missing_folders([folder])

        # Save ensemble settings
        logger.debug("Saving ensemble setup to %s/ensemble.json", folder)
        settings = {"estimator_type": self.estimator_type, "n_estimators": self.n_estimators}

        with open(folder + "/ensemble.json", "w") as f:
            json.dump(settings, f)

        # Save estimators
        for i, estimator in enumerate(self.estimators):
            estimator.save(folder + "/estimator_" + str(i), save_model=save_model)

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
        logger.debug("Loading ensemble setup from %s/ensemble.json", folder)
        with open(folder + "/ensemble.json", "r") as f:
            settings = json.load(f)

        self.n_estimators = int(settings["n_estimators"])
        try:
            estimator_type = str(settings["estimator_type"])
        except KeyError:
            raise RuntimeError(
                "Can't find estimator type information in file. Maybe this file was created with"
                " an incompatible MadMiner version < v0.3.0?"
            )
        logger.info("Found %s ensemble with %s estimators", estimator_type, self.n_estimators)

        # Load estimators
        self.estimators = []
        for i in range(self.n_estimators):
            estimator = self._get_estimator_class(estimator_type)()
            estimator.load(folder + "/estimator_" + str(i))
            self.estimators.append(estimator)
        self._check_consistency()

    def _check_consistency(self):
        """
        Internal function that checks if all estimators belong to the same category
        (local score regression, single-parameterized likelihood ratio estimator,
        doubly parameterized likelihood ratio estimator).

        Raises
        ------
        RuntimeError
            Estimators are inconsistent.

        """
        # Accumulate methods of all estimators
        all_types = [self._get_estimator_type(estimator) for estimator in self.estimators]
        all_n_parameters = [estimator.n_parameters for estimator in self.estimators]
        all_n_observables = [estimator.n_observables for estimator in self.estimators]

        # Check consistency of methods
        self.estimator_type = None

        for estimator_type in all_types:
            if self.estimator_type is None:
                self.estimator_type = estimator_type

            if self.estimator_type != estimator_type:
                raise RuntimeError(
                    "Ensemble with inconsistent estimator methods! All methods have to be either"
                    " single-parameterized ratio estimators, doubly parameterized ratio estimators,"
                    " or local score estimators. Found types " + ", ".join(all_types) + "."
                )

        # Check consistency of parameter and observable numnbers
        self.n_parameters = None
        self.n_observables = None

        for estimator_n_parameters, estimator_n_observables in zip(all_n_parameters, all_n_observables):
            if self.n_parameters is None:
                self.n_parameters = estimator_n_parameters
            if self.n_observables is None:
                self.n_observables = estimator_n_observables

            if self.n_parameters is not None and self.n_parameters != estimator_n_parameters:
                raise RuntimeError(
                    "Ensemble with inconsistent numbers of parameters for different estimators: %s", all_n_parameters
                )
            if self.n_observables is not None and self.n_observables != estimator_n_observables:
                raise RuntimeError(
                    "Ensemble with inconsistent numbers of parameters for different estimators: %s", all_n_observables
                )

    @staticmethod
    def _get_estimator_type(estimator):
        if not isinstance(estimator, Estimator):
            raise RuntimeError("Estimator is not an Estimator instance!")

        if isinstance(estimator, ParameterizedRatioEstimator):
            return "parameterized_ratio"
        elif isinstance(estimator, DoubleParameterizedRatioEstimator):
            return "double_parameterized_ratio"
        elif isinstance(estimator, ScoreEstimator):
            return "score"
        elif isinstance(estimator, LikelihoodEstimator):
            return "likelihood"
        else:
            raise RuntimeError("Estimator is an unknown Estimator type!")

    @staticmethod
    def _get_estimator_class(estimator_type):
        if estimator_type == "parameterized_ratio":
            return ParameterizedRatioEstimator
        elif estimator_type == "double_parameterized_ratio":
            return DoubleParameterizedRatioEstimator
        elif estimator_type == "score":
            return ScoreEstimator
        elif estimator_type == "likelihood":
            return LikelihoodEstimator
        else:
            raise RuntimeError("Unknown estimator type {}!".format(estimator_type))


def load_estimator(filename):
    if os.path.isdir(filename):
        model = Ensemble()
        model.load(filename)

    else:
        with open(filename + "_settings.json", "r") as f:
            settings = json.load(f)
        try:
            estimator_type = settings["estimator_type"]
        except KeyError:
            raise RuntimeError("Undefined estimator type")

        if estimator_type == "parameterized_ratio":
            model = ParameterizedRatioEstimator()
        elif estimator_type == "double_parameterized_ratio":
            model = DoubleParameterizedRatioEstimator()
        elif estimator_type == "score":
            model = ScoreEstimator()
        elif estimator_type == "likelihood":
            model = LikelihoodEstimator()
        else:
            raise RuntimeError("Unknown estimator type {}!".format(estimator_type))

        model.load(filename)

    return model


class TheresAGoodReasonThisDoesntWork(Exception):
    pass
