import json
import logging
import numpy as np
import torch

from abc import ABC
from abc import abstractmethod
from pathlib import Path

from ..utils.various import load_and_check


logger = logging.getLogger(__name__)


class Estimator(ABC):
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

    def __init__(self, features=None, n_hidden=(100,), activation="tanh", dropout_prob=0.0):
        self.features = features
        self.n_hidden = n_hidden
        self.activation = activation
        self.dropout_prob = dropout_prob

        self.model = None
        self.n_observables = None
        self.n_parameters = None
        self.x_scaling_means = None
        self.x_scaling_stds = None

    @abstractmethod
    def _create_model(self):
        raise NotImplementedError()

    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def evaluate_log_likelihood(self, *args, **kwargs):
        """
        Log likelihood estimation. Signature depends on the type of estimator. The first returned value is the log
        likelihood with shape `(n_thetas, n_x)`.
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate_log_likelihood_ratio(self, *args, **kwargs):
        """
        Log likelihood ratio estimation. Signature depends on the type of estimator. The first returned value is the log
        likelihood ratio with shape `(n_thetas, n_x)` or `(n_x)`.
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate_score(self, *args, **kwargs):
        """
        Score estimation. Signature depends on the type of estimator. The only returned value is the score with shape
        `(n_x)`.
        """
        raise NotImplementedError()

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
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        # Save settings
        logger.debug("Saving settings to %s_settings.json", filename)

        settings = self._wrap_settings()

        with open(f"{filename}_settings.json", "w") as f:
            json.dump(settings, f)

        # Save scaling
        if self.x_scaling_stds is not None and self.x_scaling_means is not None:
            logger.debug("Saving input scaling information to %s_x_means.npy and %s_x_stds.npy", filename, filename)
            np.save(f"{filename}_x_means.npy", self.x_scaling_means)
            np.save(f"{filename}_x_stds.npy", self.x_scaling_stds)

        # Save state dict
        logger.debug("Saving state dictionary to %s_state_dict.pt", filename)
        torch.save(self.model.state_dict(), f"{filename}_state_dict.pt")

        # Save model
        if save_model:
            logger.debug("Saving model to %s_model.pt", filename)
            torch.save(self.model, f"{filename}_model.pt")

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
        with open(f"{filename}_settings.json", "r") as f:
            settings = json.load(f)
        self._unwrap_settings(settings)
        self._create_model()

        # Load scaling
        try:
            self.x_scaling_means = np.load(f"{filename}_x_means.npy")
            self.x_scaling_stds = np.load(f"{filename}_x_stds.npy")
            logger.debug(
                "  Found input scaling information: means %s, stds %s", self.x_scaling_means, self.x_scaling_stds
            )
        except FileNotFoundError:
            logger.warning("Scaling information not found in %s", filename)
            self.x_scaling_means = None
            self.x_scaling_stds = None

        # Load state dict
        logger.debug("Loading state dictionary from %s_state_dict.pt", filename)
        self.model.load_state_dict(torch.load(f"{filename}_state_dict.pt", map_location="cpu"))

    def initialize_input_transform(self, x, transform=True, overwrite=True):
        if self.x_scaling_stds is not None and self.x_scaling_means is not None and not overwrite:
            logger.info(
                "Input rescaling already defined. To overwrite, call initialize_input_transform(x, overwrite=True)."
            )
        elif transform:
            logger.info("Setting up input rescaling")
            self.x_scaling_means = np.mean(x, axis=0)
            self.x_scaling_stds = np.maximum(np.std(x, axis=0), 1.0e-6)
        else:
            logger.info("Disabling input rescaling")
            n_parameters = x.shape[0]

            self.x_scaling_means = np.zeros(n_parameters)
            self.x_scaling_stds = np.ones(n_parameters)

    def _transform_inputs(self, x):
        if self.x_scaling_means is not None and self.x_scaling_stds is not None:
            if isinstance(x, torch.Tensor):
                x_scaled = x - torch.tensor(self.x_scaling_means, dtype=x.dtype, device=x.device)
                x_scaled = x_scaled / torch.tensor(self.x_scaling_stds, dtype=x.dtype, device=x.device)
            else:
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
            "dropout_prob": self.dropout_prob,
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
            self.features = [int(item) for item in self.features]

        try:
            self.dropout_prob = float(settings["dropout_prob"])
        except KeyError:
            self.dropout_prob = 0.0
            logger.info(
                "Can't find dropout probability in model file. Probably this file was created with an older"
                " MadMiner version < 0.6.1. That's totally fine, we'll just stick to the default of 0 (no"
                " dropout)."
            )

    def calculate_fisher_information(self, x, theta=None, weights=None, n_events=1, sum_events=True):
        """
        Calculates the expected Fisher information matrix based on the kinematic information in a given number of
        events.

        Parameters
        ----------
        x : str or ndarray
            Sample of observations, or path to numpy file with observations. Note that this sample has to be sampled
            from the reference parameter where the score is estimated with the SALLY / SALLINO estimator.

        theta: None or ndarray
            Numerator parameter point, or filename of a pickled numpy array. Has no effect for ScoreEstimator.

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

        # Estimate scores
        t_hats = self.evaluate_score(x=x, theta=np.array([theta for _ in x]), nuisance_mode="keep")

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


class ConditionalEstimator(Estimator, ABC):
    """
    Abstract class for estimator that is conditional on theta. Subclassed by ParameterizedRatioEstimator,
    DoubleParameterizedRatioEstimator, and LikelihoodEstimator (but not ScoreEstimator).

    Adds functionality to rescale parameters.
    """

    def __init__(self, features=None, n_hidden=(100,), activation="tanh", dropout_prob=0.0):
        super().__init__(features, n_hidden, activation, dropout_prob)

        self.theta_scaling_means = None
        self.theta_scaling_stds = None

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

        super().save(filename, save_model)

        # Save param scaling
        if self.theta_scaling_stds is not None and self.theta_scaling_means is not None:
            logger.debug(
                "Saving parameter scaling information to %s_theta_means.npy and %s_theta_stds.npy", filename, filename
            )
            np.save(f"{filename}_theta_means.npy", self.theta_scaling_means)
            np.save(f"{filename}_theta_stds.npy", self.theta_scaling_stds)

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

        super().load(filename)

        # Load param scaling
        try:
            self.theta_scaling_means = np.load(f"{filename}_theta_means.npy")
            self.theta_scaling_stds = np.load(f"{filename}_theta_stds.npy")
            logger.debug(
                "  Found parameter scaling information: means %s, stds %s",
                self.theta_scaling_means,
                self.theta_scaling_stds,
            )
        except FileNotFoundError:
            logger.warning("Parameter scaling information not found in %s", filename)
            self.theta_scaling_means = None
            self.theta_scaling_stds = None

    def initialize_parameter_transform(self, theta, transform=True, overwrite=True):
        if self.x_scaling_stds is not None and self.x_scaling_means is not None and not overwrite:
            logger.info(
                "Parameter rescaling already defined. To overwrite, call initialize_parameter_transform(theta, overwrite=True)."
            )
        elif transform:
            logger.info("Setting up parameter rescaling")
            self.theta_scaling_means = np.mean(theta, axis=0)
            self.theta_scaling_stds = np.maximum(np.std(theta, axis=0), 1.0e-6)
        else:
            logger.info("Disabling parameter rescaling")
            self.theta_scaling_means = None
            self.theta_scaling_stds = None

    def _transform_parameters(self, theta):
        if self.theta_scaling_means is not None and self.theta_scaling_stds is not None:
            if isinstance(theta, torch.Tensor):
                theta_scaled = theta - torch.tensor(self.theta_scaling_means, dtype=theta.dtype, device=theta.device)
                theta_scaled = theta_scaled / torch.tensor(
                    self.theta_scaling_stds, dtype=theta.dtype, device=theta.device
                )
            else:
                theta_scaled = theta - self.theta_scaling_means[np.newaxis, :]
                theta_scaled /= self.theta_scaling_stds[np.newaxis, :]
        else:
            theta_scaled = theta
        return theta_scaled

    def _transform_score(self, t_xz, inverse=False):
        if self.theta_scaling_means is not None and self.theta_scaling_stds is not None and t_xz is not None:
            if inverse:
                t_xz_scaled = t_xz / self.theta_scaling_stds[np.newaxis, :]
            else:
                t_xz_scaled = t_xz * self.theta_scaling_stds[np.newaxis, :]
        else:
            t_xz_scaled = t_xz
        return t_xz_scaled


class TheresAGoodReasonThisDoesntWork(Exception):
    pass
