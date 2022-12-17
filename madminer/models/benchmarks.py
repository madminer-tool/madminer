from dataclasses import dataclass
from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Iterable


@dataclass
class Benchmark:

    name: str
    values: Dict[str, float]
    is_nuisance: bool = False
    is_reference: bool = False

    @classmethod
    def from_params(cls, name: str, param_names: Iterable[str], param_values: Iterable[float]):
        """
        Creates an instance from lists of parameter names and values

        Parameters
        ----------
        name: str
        param_names: list
        param_values: list

        Returns
        -------
            Benchmark
        """

        return cls(
            name=name,
            values=OrderedDict(zip(param_names, param_values)),
        )

    def __str__(self) -> str:
        """
        Formats the benchmark in a nice way

        Returns
        -------
            Formatted benchmark string
        """

        return f"{self.name}: {', '.join(self._format_values())}"

    def _format_values(self, precision: int = 2) -> List[str]:
        """
        Formats the dictionary of parameter values

        Parameters
        ----------
        precision: int (optional)
            Precision to be used when displaying the values. Default = 2

        Returns
        -------
        formatted_pairs: list
            List of string formatted name - value pairs
        """

        # fmt: off
        lower_limit = (10.00 ** -precision) * 2.0
        upper_limit = (10.00 ** +precision) * 1.0
        formatted_pairs = []
        # fmt: on

        for parameter_name, parameter_value in self.values.items():
            if lower_limit < parameter_value < upper_limit:
                template = f"{{0:.{precision}f}}"
            else:
                template = f"{{0:.{precision}e}}"

            formatted_pairs.append(f"{parameter_name} = {template.format(parameter_value)}")

        return formatted_pairs


@dataclass
class FiniteDiffBenchmark:

    base_name: str
    shift_names: Dict[str, str]

    @classmethod
    def from_params(cls, base_name: str, param_names: Iterable[str], shift_names: Iterable[str]):
        """
        Creates an instance from lists of parameter names and shift names

        Parameters
        ----------
        base_name: str
        param_names: list
        shift_names: list

        Returns
        -------
            Benchmark
        """

        return cls(
            base_name=base_name,
            shift_names=OrderedDict(zip(param_names, shift_names)),
        )
