import logging

import numpy as np

from .parameterized_ratio import ParameterizedRatioEstimator
from ..utils.interfaces.hdf5 import load_madminer_settings
from ..utils.ml.models.ratio import DenseMorphingAwareRatioModel
from ..utils.ml.models.ratio import DenseQuadraticMorphingAwareRatioModel
from ..utils.morphing import PhysicsMorpher

logger = logging.getLogger(__name__)


class MorphingAwareRatioEstimator(ParameterizedRatioEstimator):
    def __init__(
        self,
        morphing_setup_filename=None,
        optimize_morphing_basis=False,
        features=None,
        n_hidden=(100,),
        activation="tanh",
        dropout_prob=0.0,
    ):
        super().__init__(features, n_hidden, activation, dropout_prob)

        if morphing_setup_filename is not None:
            self.components, self.morphing_matrix = self._load_morphing_setup(
                morphing_setup_filename, optimize_morphing_basis
            )
            logger.info("Setting up morphing-aware ratio estimator with %s morphing components", len(self.components))
        else:
            self.components, self.morphing_matrix = None, None

    def train(self, *args, **kwargs):
        if self.morphing_matrix is None:
            raise RuntimeError(
                "Please provide morphing setup during instantiation or load a previously trained morphing-aware "
                "estimator!"
            )

        super().train(*args, scale_parameters=False, **kwargs)

    def _load_morphing_setup(self, filename, optimize_morphing_basis=False):
        (
            parameters,
            _,
            _,
            morphing_components,
            morphing_matrix,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = load_madminer_settings(filename, include_nuisance_benchmarks=False)

        if optimize_morphing_basis:
            logger.info("Optimizing morphing basis for morphing-aware estimator")
            morpher = PhysicsMorpher(parameters_from_madminer=parameters)
            morpher.use_madminer_interface = False
            morpher.set_components(morphing_components)
            basis = morpher.optimize_basis(n_trials=1000, n_test_thetas=1000)
            logger.info("Found morphing basis:")
            for i, theta in enumerate(basis, start=1):
                logger.info("  Basis vector %s: %s", i, theta)
            morphing_matrix = morpher.calculate_morphing_matrix(basis)

        return morphing_components, morphing_matrix

    def _create_model(self):
        self.model = DenseMorphingAwareRatioModel(
            components=self.components,
            morphing_matrix=self.morphing_matrix,
            n_observables=self.n_observables,
            n_parameters=self.n_parameters,
            n_hidden=self.n_hidden,
            activation=self.activation,
            dropout_prob=self.dropout_prob,
        )

    def _wrap_settings(self):
        settings = super()._wrap_settings()
        settings["components"] = self.components.tolist()
        settings["morphing_matrix"] = self.morphing_matrix.tolist()
        return settings

    def _unwrap_settings(self, settings):
        super()._unwrap_settings(settings)

        self.components = np.array(settings["components"])
        self.morphing_matrix = np.array(settings["morphing_matrix"])


class QuadraticMorphingAwareRatioEstimator(ParameterizedRatioEstimator):
    """
    Specific morphing-aware likelihood ratio estimator for a single parameter and theta1 = 0.

    Uses the quadratic parameterization of 2007.10356: r_hat(x, theta) = (1 + theta A(x))^2 + (theta B(x))^2.
    """

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs, scale_parameters=False)

    def _create_model(self):
        self.model = DenseQuadraticMorphingAwareRatioModel(
            n_observables=self.n_observables,
            n_parameters=self.n_parameters,
            n_hidden=self.n_hidden,
            activation=self.activation,
            dropout_prob=self.dropout_prob,
        )
