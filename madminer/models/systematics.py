from dataclasses import dataclass
from enum import Enum
from typing import Union


SystematicValue = Union[str, float]


class SystematicType(str, Enum):
    """
    Type of the Systematic value:

    - "pdf": the nuisance parameters effect will be determined by varying the PDF used.
    - "norm": it will affect the overall normalization of one or multiple samples in the process.
    - "scale": the nuisance parameter effect will be determined by varying factorization or regularization scales.
    """

    PDF = "pdf"
    NORM = "norm"
    SCALE = "scale"

    @classmethod
    def from_str(cls, value):
        for k, v in cls.__members__.items():
            if v == value:
                return cls[k]

        raise ValueError(f"Invalid systematic type: {value}")


class SystematicScale(str, Enum):
    """
    Scale of Systematic value:

    - "mu": sets both factorization and regularization scales vary
    - "muf": sets that only the factorization scale varies
    - "mur": sets that only the regularization scale varies
    """

    MU = "mu"
    MUF = "muf"
    MUR = "mur"

    @classmethod
    def from_str(cls, value):
        for k, v in cls.__members__.items():
            if v == value:
                return cls[k]

        raise ValueError(f"Invalid systematic scale: {value}")


@dataclass
class Systematic:

    name: str
    type: SystematicType
    value: SystematicValue
    scale: SystematicScale = None

    def __str__(self) -> str:
        """
        Formats the systematic in a nice way

        Returns
        -------
            Formatted systematic string
        """

        return (
            f"{self.name} "
            f"("
            f"Type: {self.type}, "
            f"Value: {self.value}, "
            f"Scale: {self.scale}"
            f")"
        )

    def __post_init__(self):
        """Perform certain attribute quality assertions"""

        if self.type != SystematicType.SCALE:
            assert self.scale is None, "Cannot specify scale"

        if self.type == SystematicType.SCALE:
            assert self.scale is not None, "Must specify scale"
