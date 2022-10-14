"""A representation of the two's complement format used for INT8."""
from dataclasses import dataclass
from functools import total_ordering
from typing import Any


@dataclass
class TwosComplementFormat:
    """Can be used to express any two's complement signed integer."""

    width: int

    @property
    def abs_min(self) -> float:
        """Absolute minimum representable value."""
        return 1

    @property
    def abs_max(self) -> float:
        """Absolute maximum representable value."""
        return float(2 ** (self.width - 1) - 1)


@dataclass
class ScaledTwosComplementFormat(TwosComplementFormat):
    """A scaled (multiply all values by `scale`) version of `TwosComplementFormat`."""

    scale: int

    @property
    def abs_min(self) -> float:
        return super().abs_min * self.scale

    @property
    def abs_max(self) -> float:
        return super().abs_max * self.scale


@dataclass
@total_ordering
class TwosComplementInstance:
    """An instance of a number using a (possibly scaled) `TwosComplementFormat`."""

    format: TwosComplementFormat
    uint: int

    def __post_init__(self) -> None:
        assert 0 <= self.uint < 2**self.format.width, self.uint

    @property
    def value(self) -> float:
        """The numerical value of the bitstring,
        as defined by the two's complement format.
        """
        if self.uint <= self.format.abs_max:
            return self.uint
        return float(self.uint - 2**self.format.width)

    def __repr__(self) -> str:
        return str(self.value)

    def __eq__(self, other: Any) -> bool:
        return self.value.__eq__(other)

    def __lt__(self, other: Any) -> bool:
        return self.value.__lt__(other)
