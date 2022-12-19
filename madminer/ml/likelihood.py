import logging

from collections import OrderedDict

import numpy as np

from .base import ConditionalEstimator
from ..utils.ml.eval import evaluate_flow_model
from ..utils.ml.models.maf import ConditionalMaskedAutoregressiveFlow
from ..utils.ml.models.maf_mog import ConditionalMixtureMaskedAutoregressiveFlow
from ..utils.ml.trainer import FlowTrainer
from ..utils.ml.utils import get_optimizer
from ..utils.ml.utils import get_loss
from ..utils.various import load_and_check
from ..utils.various import shuffle
from ..utils.various import restrict_samplesize

logger = logging.getLogger(__name__)


class LikelihoodEstimator(ConditionalEstimator):
    """A neural estimator of the density or likelihood evaluated at a reference hypothesis as a function
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

    batch_norm : None or float, optional
        If not None, batch normalization is used, where this value sets the alpha parameter in the calculation
        of the running average of the mean and variance. Default value: None.
    """

    def __init__(self, features=None, n_components=1, n_mades=5, n_hidden=(100,), activation="tanh", batch_norm=None):
        super().__init__(features, n_hidden, activation, dropout_prob=0.0)

        self.n_components = n_components
        self.n_mades = n_mades
        self.batch_norm = batch_norm

    def train(
        self,
        method,
        x,
        theta,
        t_xz=None,
        x_val=None,
        theta_val=None,
        t_xz_val=None,
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
        scale_parameters=True,
        n_workers=8,
        clip_gradient=None,
        early_stopping_patience=None,
    ):
        """
        Trains the network.

        Parameters
        ----------
        method : str
            The inference method used for training. Allowed values are 'nde' and 'scandal'.

        x : ndarray or str
            Observations, or filename of a pickled numpy array.

        theta : ndarray or str
            Numerator parameter point, or filename of a pickled numpy array.

        t_xz : ndarray or str or None, optional
            Joint scores at theta, or filename of a pickled numpy array. Default value: None.

        x_val : ndarray or str or None, optional
            Validation observations, or filename of a pickled numpy array. If None
            and validation_split > 0, validation data will be randomly selected from the training data.
            Default value: None.

        theta_val : ndarray or str or None, optional
            Validation numerator parameter points, or filename of a pickled numpy array. If None
            and validation_split > 0, validation data will be randomly selected from the training data.
            Default value: None.

        t_xz_val : ndarray or str or None, optional
            Validation joint scores at theta, or filename of a pickled numpy array. If None
            and validation_split > 0, validation data will be randomly selected from the training data.
            Default value: None.

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

        scale_parameters : bool, optional
            Whether parameters are rescaled to mean zero and unit variance before going into the neural network.
            Default value: True.

        Returns
        -------
        result: ndarray
            Training and validation losses from FlowTrainer.train
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

        # Validation data
        external_validation = x_val is not None and theta_val is not None
        if external_validation:
            theta_val = load_and_check(theta_val, memmap_files_larger_than_gb=memmap_threshold)
            x_val = load_and_check(x_val, memmap_files_larger_than_gb=memmap_threshold)
            t_xz_val = load_and_check(t_xz_val, memmap_files_larger_than_gb=memmap_threshold)

            logger.info("Found %s separate validation samples", x_val.shape[0])

            assert x_val.shape[1] == n_observables
            assert theta_val.shape[1] == n_parameters
            if t_xz is not None:
                assert t_xz_val is not None, "When providing t_xz and sep. validation data, also provide t_xz_val"

        # Scale features
        if scale_inputs:
            self.initialize_input_transform(x, overwrite=False)
            x = self._transform_inputs(x)
            if external_validation:
                x_val = self._transform_inputs(x_val)
        else:
            self.initialize_input_transform(x, False, overwrite=False)

        # Scale parameters
        if scale_parameters:
            logger.info("Rescaling parameters")
            self.initialize_parameter_transform(theta)
            theta = self._transform_parameters(theta)
            t_xz = self._transform_score(t_xz, inverse=False)
            if external_validation:
                t_xz_val = self._transform_score(t_xz_val, inverse=False)
        else:
            self.initialize_parameter_transform(theta, False)

        # Shuffle labels
        if shuffle_labels:
            logger.info("Shuffling labels")
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
        data = self._package_training_data(method, x, theta, t_xz)
        if external_validation:
            data_val = self._package_training_data(method, x_val, theta_val, t_xz_val)
        else:
            data_val = None

        # Create model
        if self.model is None:
            self._create_model()

        # Losses
        loss_functions, loss_labels, loss_weights = get_loss(method, alpha)

        # Optimizer
        opt, opt_kwargs = get_optimizer(optimizer, nesterov_momentum)

        # Train model
        logger.info("Training model")
        trainer = FlowTrainer(self.model, n_workers=n_workers)
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
            logger.debug("Loading evaluation data")
        theta = load_and_check(theta)
        x = load_and_check(x)

        # Scale observables
        x = self._transform_inputs(x)

        # Restrict features
        if self.features is not None:
            x = x[:, self.features]

        # Evaluation for all other methods
        all_log_p_hat = []
        all_t_hat = []

        if test_all_combinations:
            logger.debug("Starting ratio evaluation for %s x-theta combinations", len(theta) * len(x))

            for i, this_theta in enumerate(theta, start=1):
                logger.debug("Starting log likelihood evaluation for thetas %s / %s: %s", i, len(theta), this_theta)

                log_p_hat, t_hat = evaluate_flow_model(
                    model=self.model,
                    thetas=[this_theta],
                    xs=x,
                    evaluate_score=evaluate_score,
                )

                all_log_p_hat.append(log_p_hat)
                all_t_hat.append(t_hat)

            all_log_p_hat = np.array(all_log_p_hat)
            all_t_hat = np.array(all_t_hat)

        else:
            logger.debug("Starting log likelihood evaluation")

            all_log_p_hat, all_t_hat = evaluate_flow_model(
                model=self.model, thetas=theta, xs=x, evaluate_score=evaluate_score
            )

        logger.debug("Evaluation done")
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
        logger.debug("Loading evaluation data")
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
            raise RuntimeError(f"Method {method} requires joint score information")

    @staticmethod
    def _package_training_data(method, x, theta, t_xz):
        data = OrderedDict()
        data["x"] = x
        data["theta"] = theta
        if method in ["scandal"]:
            data["t_xz"] = t_xz
        return data

    def _wrap_settings(self):
        settings = super()._wrap_settings()
        settings["estimator_type"] = "likelihood"
        settings["n_components"] = self.n_components
        settings["batch_norm"] = self.batch_norm
        settings["n_mades"] = self.n_mades
        return settings

    def _unwrap_settings(self, settings):
        super()._unwrap_settings(settings)

        estimator_type = str(settings["estimator_type"])
        if estimator_type != "likelihood":
            raise RuntimeError(f"Saved model is an incompatible estimator type {estimator_type}.")

        self.n_components = int(settings["n_components"])
        self.n_mades = int(settings["n_mades"])
        self.batch_norm = settings["batch_norm"]
        if self.batch_norm == "None":
            self.batch_norm = None
        if self.batch_norm is not None:
            self.batch_norm = float(self.batch_norm)
