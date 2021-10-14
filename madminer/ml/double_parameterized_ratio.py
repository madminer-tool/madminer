import logging
import numpy as np

from collections import OrderedDict

from .base import ConditionalEstimator, TheresAGoodReasonThisDoesntWork
from ..utils.ml.eval import evaluate_ratio_model
from ..utils.ml.models.ratio import DenseDoublyParameterizedRatioModel
from ..utils.ml.trainer import DoubleParameterizedRatioTrainer
from ..utils.ml.utils import get_optimizer, get_loss
from ..utils.various import load_and_check, shuffle, restrict_samplesize


logger = logging.getLogger(__name__)


class DoubleParameterizedRatioEstimator(ConditionalEstimator):
    """
    A neural estimator of the likelihood ratio as a function of the observation x,
    the numerator hypothesis theta0, and the denominator hypothesis theta1.
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
        x_val=None,
        y_val=None,
        theta0_val=None,
        theta1_val=None,
        r_xz_val=None,
        t_xz0_val=None,
        t_xz1_val=None,
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
            The inference method used for training. Allowed values are 'alice', 'alices', 'carl', 'cascal', 'rascal',
            and 'rolr'.

        x : ndarray or str
            Observations, or filename of a pickled numpy array.

        y : ndarray or str
            Class labels (0 = numerator, 1 = denominator), or filename of a pickled numpy array.

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

        x_val : ndarray or str or None, optional
            Validation observations, or filename of a pickled numpy array. If None
            and validation_split > 0, validation data will be randomly selected from the training data.
            Default value: None.

        y_val : ndarray or str or None, optional
            Validation labels (0 = numerator, 1 = denominator), or filename of a pickled numpy array. If None
            and validation_split > 0, validation data will be randomly selected from the training data.
            Default value: None.

        theta0_val : ndarray or str or None, optional
            Validation numerator parameter points, or filename of a pickled numpy array. If None
            and validation_split > 0, validation data will be randomly selected from the training data.
            Default value: None.

        theta1_val : ndarray or str or None, optional
            Validation denominator parameter points, or filename of a pickled numpy array. If None
            and validation_split > 0, validation data will be randomly selected from the training data.
            Default value: None.

        r_xz_val : ndarray or str or None, optional
            Validation joint likelihood ratio, or filename of a pickled numpy array. If None
            and validation_split > 0, validation data will be randomly selected from the training data.
            Default value: None.

        t_xz0_val : ndarray or str or None, optional
            Validation joint scores at theta0, or filename of a pickled numpy array. If None
            and validation_split > 0, validation data will be randomly selected from the training data.
            Default value: None.

        t_xz1_val : ndarray or str or None, optional
            Validation joint scores at theta1, or filename of a pickled numpy array. If None
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

        Returns
        -------
        result: ndarray
            Training and validation losses from DoubleParameterizedRatioTrainer.train
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

        # Validation data
        external_validation = (
            x_val is not None and y_val is not None and theta0_val is not None and theta1_val is not None
        )
        if external_validation:
            theta0_val = load_and_check(theta0_val, memmap_files_larger_than_gb=memmap_threshold)
            theta1_val = load_and_check(theta1_val, memmap_files_larger_than_gb=memmap_threshold)
            x_val = load_and_check(x_val, memmap_files_larger_than_gb=memmap_threshold)
            y_val = load_and_check(y_val, memmap_files_larger_than_gb=memmap_threshold)
            r_xz_val = load_and_check(r_xz_val, memmap_files_larger_than_gb=memmap_threshold)
            t_xz0_val = load_and_check(t_xz0_val, memmap_files_larger_than_gb=memmap_threshold)
            t_xz1_val = load_and_check(t_xz1_val, memmap_files_larger_than_gb=memmap_threshold)

            logger.info("Found %s separate validation samples", x_val.shape[0])

            assert x_val.shape[1] == n_observables
            assert theta0_val.shape[1] == n_parameters
            assert theta1_val.shape[1] == n_parameters

            if r_xz is not None:
                assert r_xz_val is not None, "When providing r_xz and sep. validation data, also provide r_xz_val"
            if t_xz0 is not None:
                assert t_xz0_val is not None, "When providing t_xz0 and sep. validation data, also provide t_xz0_val"
            if t_xz1 is not None:
                assert t_xz1_val is not None, "When providing t_xz1 and sep. validation data, also provide t_xz1_val"

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
            self.initialize_parameter_transform(np.concatenate((theta0, theta1), 0))
            theta0 = self._transform_parameters(theta0)
            theta1 = self._transform_parameters(theta1)
            t_xz0 = self._transform_score(t_xz0, inverse=False)
            t_xz1 = self._transform_score(t_xz1, inverse=False)
            if external_validation:
                t_xz0_val = self._transform_score(t_xz0_val, inverse=False)
                t_xz1_val = self._transform_score(t_xz1_val, inverse=False)
        else:
            self.initialize_parameter_transform(np.concatenate((theta0, theta1), 0), False)

        # Shuffle labels
        if shuffle_labels:
            logger.info("Shuffling labels")
            y, r_xz, t_xz0, t_xz1 = shuffle(y, r_xz, t_xz0, t_xz1)

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
        data = self._package_training_data(method, x, theta0, theta1, y, r_xz, t_xz0, t_xz1)

        if external_validation:
            data_val = self._package_training_data(
                method,
                x_val,
                theta0_val,
                theta1_val,
                y_val,
                r_xz_val,
                t_xz0_val,
                t_xz1_val,
            )
        else:
            data_val = None

        # Create model
        if self.model is None:
            logger.info("Creating model")
            self._create_model()

        # Losses
        loss_functions, loss_labels, loss_weights = get_loss(method + "2", alpha)

        # Optimizer
        opt, opt_kwargs = get_optimizer(optimizer, nesterov_momentum)

        # Train model
        logger.info("Training model")
        trainer = DoubleParameterizedRatioTrainer(self.model, n_workers=n_workers)
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

        all_log_r_hat = []
        all_t_hat0 = []
        all_t_hat1 = []

        if test_all_combinations:
            logger.debug("Starting ratio evaluation for %s x-theta combinations", len(theta0) * len(x))

            for i, (this_theta0, this_theta1) in enumerate(zip(theta0, theta1), start=1):
                logger.debug(
                    "Starting ratio evaluation for thetas %s / %s: %s vs %s",
                    i,
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
            logger.debug("Starting ratio evaluation")
            _, all_log_r_hat, all_t_hat0, all_t_hat1 = evaluate_ratio_model(
                model=self.model,
                method_type="double_parameterized_ratio",
                theta0s=theta0,
                theta1s=theta1,
                xs=x,
                evaluate_score=evaluate_score,
            )

        logger.debug("Evaluation done")
        return all_log_r_hat, all_t_hat0, all_t_hat1

    def evaluate_log_likelihood(self, *args, **kwargs):
        raise TheresAGoodReasonThisDoesntWork(
            "This estimator can only estimate likelihood ratios, not the likelihood itself!"
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
            dropout_prob=self.dropout_prob,
        )

    @staticmethod
    def _check_required_data(method, r_xz, t_xz0, t_xz1):
        if method in ["cascal", "alices", "rascal"] and (t_xz0 is None or t_xz1 is None):
            raise RuntimeError(f"Method {method} requires joint score information")
        if method in ["rolr", "alice", "alices", "rascal"] and r_xz is None:
            raise RuntimeError(f"Method {method} requires joint likelihood ratio information")

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
        settings = super()._wrap_settings()
        settings["estimator_type"] = "double_parameterized_ratio"
        return settings

    def _unwrap_settings(self, settings):
        super()._unwrap_settings(settings)

        estimator_type = str(settings["estimator_type"])
        if estimator_type != "double_parameterized_ratio":
            raise RuntimeError(f"Saved model is an incompatible estimator type {estimator_type}.")
