import json

from pathlib import Path

from .ensemble import Ensemble
from .double_parameterized_ratio import DoubleParameterizedRatioEstimator
from .likelihood import LikelihoodEstimator
from .parameterized_ratio import ParameterizedRatioEstimator
from .score import ScoreEstimator


def load_estimator(filename):
    if Path(filename).is_dir():
        model = Ensemble()
        model.load(filename)

    else:
        with open(f"{filename}_settings.json", "r") as f:
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
            raise RuntimeError(f"Unknown estimator type {estimator_type}!")

        model.load(filename)

    return model
