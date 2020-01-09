from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
from scipy.stats import chi2, poisson

from madminer.analysis import DataAnalyzer
from madminer.utils.various import mdot, less_logging
from madminer.ml import ParameterizedRatioEstimator, Ensemble, ScoreEstimator, LikelihoodEstimator, load_estimator
from madminer.utils.histo import Histo
from madminer.sampling import SampleAugmenter
from madminer import sampling

logger = logging.getLogger(__name__)


class LHCLikelihood(DataAnalyzer):
    def create_observed_log_likelihood_function(
        self, model_file, x_observed, include_xsec=True, luminosity=300000.0, use_torch=True
    ):
        raise NotImplementedError

    def create_expected_log_likelihood_function(
        self, model_file, x_observed, include_xsec=True, luminosity=300000.0, use_torch=True
    ):
        raise NotImplementedError
