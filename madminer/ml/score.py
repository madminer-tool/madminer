import logging

from collections import OrderedDict

import numpy as np

from .base import Estimator
from .base import TheresAGoodReasonThisDoesntWork
from ..utils.ml.eval import evaluate_local_score_model
from ..utils.ml.models.score import DenseLocalScoreModel
from ..utils.ml.trainer import LocalScoreTrainer
from ..utils.ml.utils import get_optimizer
from ..utils.ml.utils import get_loss
from ..utils.various import load_and_check
from ..utils.various import shuffle
from ..utils.various import restrict_samplesize
from ..utils.various import separate_information_blocks

logger = logging.getLogger(__name__)


class ScoreEstimator(Estimator):
    """A neural estimator of the score evaluated at a fixed reference hypothesis as a function of the
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

    def __init__(self, features=None, n_hidden=(100,), activation="tanh", dropout_prob=0.0):
        super().__init__(features, n_hidden, activation, dropout_prob)

        self.nuisance_profile_matrix = None
        self.nuisance_project_matrix = None
        self.nuisance_mode_default = "keep"

    def train(
        self,
        method,
        x,
        t_xz,
        x_val=None,
        t_xz_val=None,
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
        n_workers=8,
        clip_gradient=None,
        early_stopping_patience=None,
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
        result: ndarray
            Training and validation losses from LocalScoreTrainer.train
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

        # Validation data
        external_validation = x_val is not None and t_xz_val is not None
        if external_validation:
            x_val = load_and_check(x_val, memmap_files_larger_than_gb=memmap_threshold)
            t_xz_val = load_and_check(t_xz_val, memmap_files_larger_than_gb=memmap_threshold)

            logger.info("Found %s separate validation samples", x_val.shape[0])

            assert x_val.shape[1] == n_observables
            assert t_xz_val.shape[1] == n_parameters

        # Scale features
        if scale_inputs:
            self.initialize_input_transform(x, overwrite=False)
            x = self._transform_inputs(x)
            if external_validation:
                x_val = self._transform_inputs(x_val)
        else:
            self.initialize_input_transform(x, False, overwrite=False)

        # Shuffle labels
        if shuffle_labels:
            logger.info("Shuffling labels")
            logger.warning("Are you sure you want this?")
            t_xz = shuffle(t_xz)

        # Features
        if self.features is not None:
            x = x[:, self.features]
            logger.info("Only using %s of %s observables", x.shape[1], n_observables)
            n_observables = x.shape[1]
            if external_validation:
                x_val = x_val[:, self.features]

        # Check consistency of input with model
        if self.n_observables is None:
            self.n_observables = n_observables
        if self.n_parameters is None:
            self.n_parameters = n_parameters

        if n_parameters != self.n_parameters:
            raise RuntimeError(f"Number of parameters does not match: {n_parameters} vs {self.n_parameters}")
        if n_observables != self.n_observables:
            raise RuntimeError(f"Number of observables does not match: {n_observables} vs {self.n_observables}")

        # Data
        data = self._package_training_data(x, t_xz)
        if external_validation:
            data_val = self._package_training_data(x_val, t_xz_val)
        else:
            data_val = None

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
        trainer = LocalScoreTrainer(self.model, n_workers=n_workers)
        result = trainer.train(
            data=data,
            data_val=data_val,
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
            clip_gradient=clip_gradient,
            early_stopping_patience=early_stopping_patience,
        )
        return result

    def set_nuisance(self, fisher_information, parameters_of_interest):
        """
        Prepares the calculation of profiled scores, see https://arxiv.org/pdf/1903.01473.pdf.

        Parameters
        ----------
        fisher_information : ndarray
            Fisher information with shape `(n_parameters, n_parameters)`.

        parameters_of_interest : list of int
            List of int, with 0 <= remaining_components[i] < n_parameters. Denotes which parameters are kept in the
            profiling, and their new order.

        Returns
        -------
            None

        """
        if fisher_information.shape != (self.n_parameters, self.n_parameters):
            raise ValueError(
                f"Fisher information has wrong shape {fisher_information.shape}. "
                f"Expected {(self.n_parameters, self.n_parameters)}"
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

    def evaluate_score(self, x, theta=None, nuisance_mode="auto"):
        """
        Evaluates the score.

        Parameters
        ----------
        x : str or ndarray
            Observations, or filename of a pickled numpy array.

        theta: None or ndarray, optional
            Has no effect for ScoreEstimator. Introduced just for conformity with other Estimators.

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
            logger.debug("Loading evaluation data")
        x = load_and_check(x)

        # Scale observables
        x = self._transform_inputs(x)

        # Restrict features
        if self.features is not None:
            x = x[:, self.features]

        # Evaluation
        logger.debug("Starting score evaluation")
        t_hat = evaluate_local_score_model(model=self.model, xs=x)

        # Treatment of nuisance parameters
        if nuisance_mode == "keep":
            logger.debug("Keeping nuisance parameter in score")

        elif nuisance_mode == "project":
            if self.nuisance_project_matrix is None:
                raise ValueError(
                    "evaluate_score() was called with nuisance_mode = project, but nuisance parameters "
                    "have not been set up yet. Please call set_nuisance() first!"
                )
            logger.debug("Projecting nuisance parameter from score")
            t_hat = np.einsum("ij,xj->xi", self.nuisance_project_matrix, t_hat)

        elif nuisance_mode == "profile":
            if self.nuisance_profile_matrix is None:
                raise ValueError(
                    "evaluate_score() was called with nuisance_mode = profile, but nuisance parameters "
                    "have not been set up yet. Please call set_nuisance() first!"
                )
            logger.debug("Profiling nuisance parameter from score")
            t_hat = np.einsum("ij,xj->xi", self.nuisance_profile_matrix, t_hat)

        else:
            raise ValueError(f"Unknown nuisance_mode {nuisance_mode}")

        return t_hat

    def evaluate_log_likelihood(self, *args, **kwargs):
        raise TheresAGoodReasonThisDoesntWork("This estimator can only estimate the score, not the likelihood!")

    def evaluate_log_likelihood_ratio(self, *args, **kwargs):
        raise TheresAGoodReasonThisDoesntWork("This estimator can only estimate the score, not the likelihood ratio!")

    def evaluate(self, *args, **kwargs):
        return self.evaluate_score(*args, **kwargs)

    def save(self, filename, save_model=False):
        super().save(filename, save_model)

        # Also save Fisher information information for profiling / projections
        if self.nuisance_profile_matrix is not None and self.nuisance_project_matrix is not None:
            logger.debug(
                "Saving nuisance profiling / projection information to %s_nuisance_profile_matrix.npy and "
                "%s_nuisance_project_matrix.npy",
                filename,
                filename,
            )
            np.save(f"{filename}_nuisance_profile_matrix.npy", self.nuisance_profile_matrix)
            np.save(f"{filename}_nuisance_project_matrix.npy", self.nuisance_project_matrix)

    def load(self, filename):
        super().load(filename)

        # Load scaling
        try:
            self.nuisance_profile_matrix = np.load(f"{filename}_nuisance_profile_matrix.npy")
            self.nuisance_project_matrix = np.load(f"{filename}_nuisance_project_matrix.npy")
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
            dropout_prob=self.dropout_prob,
        )

    @staticmethod
    def _package_training_data(x, t_xz):
        data = OrderedDict()
        data["x"] = x
        data["t_xz"] = t_xz
        return data

    def _wrap_settings(self):
        settings = super()._wrap_settings()
        settings["estimator_type"] = "score"
        settings["nuisance_mode_default"] = self.nuisance_mode_default
        return settings

    def _unwrap_settings(self, settings):
        super()._unwrap_settings(settings)

        estimator_type = str(settings["estimator_type"])
        if estimator_type != "score":
            raise RuntimeError(f"Saved model is an incompatible estimator type {estimator_type}.")

        try:
            self.nuisance_mode_default = str(settings["nuisance_mode_default"])
        except KeyError:
            self.nuisance_mode_default = "keep"
            logger.warning("Did not find entry nuisance_mode_default in saved model, using default 'keep'.")
