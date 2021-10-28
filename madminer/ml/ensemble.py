import json
import logging
import numpy as np

from madminer.utils.various import load_and_check
from pathlib import Path

from .base import Estimator
from .double_parameterized_ratio import DoubleParameterizedRatioEstimator
from .likelihood import LikelihoodEstimator
from .parameterized_ratio import ParameterizedRatioEstimator
from .score import ScoreEstimator


logger = logging.getLogger(__name__)


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
                    raise ValueError(f"Estimator {estimator} is not an Estimator instance")

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
            raise ValueError(f"Estimator {estimator} is not an Estimator instance")

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
        result: ndarray
            Training and validation losses from estimator training
            
        """

        result=self.estimators[i].train(**kwargs)
        return result

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
        result_list: list of ndarray
            List of training and validation losses from estimator training
        """

        logger.info("Training %s estimators in ensemble", self.n_estimators)

        result_list = []

        for key, value in kwargs.items():
            if not isinstance(value, list):
                kwargs[key] = [value for _ in range(self.n_estimators)]

            assert len(kwargs[key]) == self.n_estimators, f"Keyword {key} has wrong length {len(value)}"

        for i, estimator in enumerate(self.estimators):
            kwargs_this_estimator = {}
            for key, value in kwargs.items():
                kwargs_this_estimator[key] = value[i]

            logger.info("Training estimator %s / %s in ensemble", i + 1, self.n_estimators)
            result = estimator.train(**kwargs_this_estimator)
            result_list.append(result)

        return result_list

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

        logger.debug("Evaluating %s estimators in ensemble", self.n_estimators)

        # Calculate weights of each estimator in vote
        if estimator_weights is None:
            estimator_weights = np.ones(self.n_estimators)
        assert len(estimator_weights) == self.n_estimators
        estimator_weights /= np.sum(estimator_weights)
        logger.debug("Estimator weights: %s", estimator_weights)

        # Calculate estimator predictions
        predictions = []
        for i, estimator in enumerate(self.estimators, start=1):
            logger.debug("Starting evaluation for estimator %s / %s in ensemble", i, self.n_estimators)
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

        logger.debug("Evaluating %s estimators in ensemble", self.n_estimators)

        # Calculate weights of each estimator in vote
        if estimator_weights is None:
            estimator_weights = np.ones(self.n_estimators)
        assert len(estimator_weights) == self.n_estimators
        estimator_weights /= np.sum(estimator_weights)
        logger.debug("Estimator weights: %s", estimator_weights)

        # Calculate estimator predictions
        predictions = []
        for i, estimator in enumerate(self.estimators, start=1):
            logger.debug("Starting evaluation for estimator %s / %s in ensemble", i, self.n_estimators)
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

        logger.debug("Evaluating %s estimators in ensemble", self.n_estimators)

        # Calculate weights of each estimator in vote
        if estimator_weights is None:
            estimator_weights = np.ones(self.n_estimators)
        assert len(estimator_weights) == self.n_estimators
        estimator_weights /= np.sum(estimator_weights)
        logger.debug("Estimator weights: %s", estimator_weights)

        # Calculate estimator predictions
        predictions = []
        for i, estimator in enumerate(self.estimators, start=1):
            logger.info("Starting evaluation for estimator %s / %s in ensemble", i, self.n_estimators)
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
        theta=None,
        obs_weights=None,
        estimator_weights=None,
        n_events=1,
        mode="score",
        calculate_covariance=True,
        sum_events=True,
        epsilon_shift=0.001,
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

        epsilon_shift : float, optional
            Small numerical factor in the error propagation. Default value: 0.001.

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
        if self.estimator_type not in ["score", "parameterized_ratio"]:
            raise NotImplementedError(
                "Fisher information calculation is only implemented for local score estimators "
                "(ScoreEstimator instances) and parameterized ratio estimators (parameterized_ratio instances)."
            )

        # Check input
        if mode not in ["score", "information", "modified_score"]:
            raise ValueError(f"Unknown mode {mode}!")

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
            for i, estimator in enumerate(self.estimators, start=1):
                logger.debug("Starting evaluation for estimator %s / %s in ensemble", i, self.n_estimators)
                predictions.append(
                    estimator.calculate_fisher_information(x=x, theta=theta, weights=obs_weights, n_events=n_events)
                )
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

        # "modified_score" mode:
        elif mode == "modified_score":
            # Load training data
            if isinstance(x, str):
                x = load_and_check(x)
            n_samples = x.shape[0]

            # Calculate score predictions
            score_predictions = []
            for i, estimator in enumerate(self.estimators, start=1):
                logger.debug("Starting evaluation for estimator %s / %s in ensemble", i, self.n_estimators)
                score_predictions.append(estimator.evaluate_score(x=x, theta=np.array([theta for _ in x])))
                logger.debug("Estimator %s predicts t(x) = %s for first event", i, score_predictions[-1][0, :])

            score_predictions = np.array(score_predictions)  # (n_estimators, n_events, n_parameters)

            # Get ensemble mean and ensemble covariance
            score_mean = np.mean(score_predictions, axis=0)  # (n_events, n_parameters)

            # For uncertainty calculation: calculate points between mean and original predictions with same mean and
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
                informations_individual = float(n_events) * np.sum(
                    obs_weights[np.newaxis, :, np.newaxis, np.newaxis]
                    * score_shifted_predictions[:, :, :, np.newaxis]
                    * score_shifted_predictions[:, :, np.newaxis, :],
                    axis=1,
                )  # (n_estimators, n_parameters, n_parameters)

                n_params = score_mean.shape[1]
                informations_individual = informations_individual.reshape(-1, n_params ** 2)
                covariance = np.cov(informations_individual.T)
                covariance = covariance.reshape(n_params, n_params, n_params, n_params)

            # Let's check the expected score
            expected_score = [np.einsum("n,ni->i", obs_weights, score_mean)]
            logger.debug("Expected per-event score (should be close to zero):\n%s", expected_score)

        # "score" mode:
        elif mode == "score":
            # Load training data
            if isinstance(x, str):
                x = load_and_check(x)
            n_samples = x.shape[0]

            # Calculate score predictions
            score_predictions = []
            for i, estimator in enumerate(self.estimators, start=1):
                logger.debug("Starting evaluation for estimator %s / %s in ensemble", i, self.n_estimators)
                score_predictions.append(estimator.evaluate_score(x=x, theta=np.array([theta for _ in x])))
                logger.debug("Estimator %s predicts t(x) = %s for first event", i, score_predictions[-1][0, :])

            score_predictions = np.array(score_predictions)  # (n_estimators, n_events, n_parameters)

            # Get ensemble mean and ensemble covariance
            score_mean = np.mean(score_predictions, axis=0)  # (n_events, n_parameters)

            # For uncertainty calculation: calculate points between mean and original predictions with same mean and
            # variance / n compared to the original predictions
            score_shifted_predictions = epsilon_shift * (score_predictions - score_mean[np.newaxis, :, :])
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
                informations_individual = float(n_events) * np.sum(
                    obs_weights[np.newaxis, :, np.newaxis, np.newaxis]
                    * score_shifted_predictions[:, :, :, np.newaxis]
                    * score_shifted_predictions[:, :, np.newaxis, :],
                    axis=1,
                )  # (n_estimators, n_parameters, n_parameters)

                n_params = score_mean.shape[1]
                informations_individual = informations_individual.reshape(-1, n_params ** 2)
                covariance = np.cov(informations_individual.T)
                covariance /= epsilon_shift ** 2
                covariance = covariance.reshape(n_params, n_params, n_params, n_params)

            # Let's check the expected score
            expected_score = [np.einsum("n,ni->i", obs_weights, score_mean)]
            logger.debug("Expected per-event score (should be close to zero):\n%s", expected_score)

        else:
            raise RuntimeError("Unknown mode %s, has to be 'information', 'score', or 'modified_score'.")

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
        Path(folder).mkdir(parents=True, exist_ok=True)

        # Save ensemble settings
        logger.debug("Saving ensemble setup to %s/ensemble.json", folder)
        settings = {"estimator_type": self.estimator_type, "n_estimators": self.n_estimators}

        with open(f"{folder}/ensemble.json", "w") as f:
            json.dump(settings, f)

        # Save estimators
        for i, estimator in enumerate(self.estimators):
            estimator.save(f"{folder}/estimator_{i}", save_model=save_model)

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
        with open(f"{folder}/ensemble.json", "r") as f:
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
            estimator.load(f"{folder}/estimator_{i}")
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
                    f"Ensemble with inconsistent estimator methods! All methods have to be either "
                    f"single-parameterized ratio estimators, doubly parameterized ratio estimators, "
                    f"or local score estimators. Found types {', '.join(all_types)}."
                )

        # Check consistency of parameter and observable numbers
        self.n_parameters = None
        self.n_observables = None

        for estimator_n_parameters, estimator_n_observables in zip(all_n_parameters, all_n_observables):
            if self.n_parameters is None:
                self.n_parameters = estimator_n_parameters
            if self.n_observables is None:
                self.n_observables = estimator_n_observables

            if self.n_parameters is not None and self.n_parameters != estimator_n_parameters:
                raise RuntimeError(
                    f"Ensemble with inconsistent numbers of parameters for different estimators: "
                    f"{all_n_parameters}"
                )
            if self.n_observables is not None and self.n_observables != estimator_n_observables:
                raise RuntimeError(
                    f"Ensemble with inconsistent numbers of observables for different estimators: "
                    f"{all_n_observables}"
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
            raise RuntimeError(f"Unknown estimator type {estimator_type}!")
