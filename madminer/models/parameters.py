from contextlib import suppress
from dataclasses import dataclass
from typing import Tuple


@dataclass
class AnalysisParameter:

    name: str
    lha_block: str
    lha_id: int
    max_power: int
    val_range: Tuple[float, float]
    transform: str

    def __str__(self) -> str:
        """
        Formats the analysis parameter in a nice way

        Returns
        -------
            Formatted parameter string
        """

        return (
            f"{self.name} "
            f"("
            f"LHA: {self.lha_block} {self.lha_id}, "
            f"Power: {self.max_power}, "
            f"Range: {self.val_range}"
            f")"
        )

    def __post_init__(self):
        """Perform certain attribute quality assertions"""

        with suppress(NameError):
            eval(self.transform)


@dataclass
class NuisanceParameter:

    name: str
    systematic: str
    benchmark_pos: str = None
    benchmark_neg: str = None

    def __str__(self) -> str:
        """
        Formats the nuisance parameter in a nice way

        Returns
        -------
            Formatted parameter string
        """

        return (
            f"{self.name} "
            f"("
            f"Systematic: {self.systematic}, "
            f"Benchmarks: {self.benchmark_pos} | {self.benchmark_neg}"
            f")"
        )
