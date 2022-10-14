"""A representation of the IEEE754 binary format, as well as variants for FP8."""
from dataclasses import dataclass
from functools import total_ordering
from typing import Any, ClassVar


@dataclass
class IEEE754BinaryFormat:
    """Can be used to express any format defined in the IEEE754 standard.

    To represent a value using this format, see `FloatInstance`.
    """

    e_width: int
    m_width: int

    inf_encoding: ClassVar[str] = "E=1s M=0s"
    nan_encoding: ClassVar[str] = "E=1s M≠0s"
    zero_encoding: ClassVar[str] = "S=0/1 E=0s M=0s"
    subnormal_encoding: ClassVar[str] = "E=0s"

    @property
    def bias(self) -> int:
        """Exponent bias."""
        return int(2 ** (self.e_width - 1)) - 1

    @property
    def min_e(self) -> int:
        """Minimum exponent.

        Note: this is not `0 - self.bias` as the 0 exponent is reserved for subnormals.
        """
        return 1 - self.bias

    @property
    def max_e(self) -> int:
        """Maximum exponent.

        Note: The `-2` accounts for the fact that the all 1s exponent denotes NaN/Inf.
        """
        return int(2**self.e_width) - 2 - self.bias

    @property
    def abs_min_normal(self) -> float:
        """Absolute minimum normal (i.e. not subnormal) representable value."""
        return float(2**self.min_e)

    @property
    def abs_min(self) -> float:
        """Absolute minimum representable value (this is in the subnormal range)."""
        return float(2 ** (self.min_e - self.m_width))

    @property
    def abs_max(self) -> float:
        """Absolute maximum representable value."""
        return float((2**self.max_e) * (2 - 2**-self.m_width))


@dataclass
class GAQProposedFormat(IEEE754BinaryFormat):
    """Used for both of Graphcore, AMD and Qualcomm's proposed FP8 formats."""

    custom_bias: int

    inf_encoding: ClassVar[str] = "N/A"
    nan_encoding: ClassVar[str] = "S=1 E=0s M=0s"
    zero_encoding: ClassVar[str] = "S=0 E=0s M=0s"

    @property
    def bias(self) -> int:
        """Exponent bias. GAQ use a bias that doesn't match the standard IEE754 one."""
        return self.custom_bias

    @property
    def max_e(self) -> int:
        """Maximum exponent. The all 1s exponent no longer denotes NaN/Inf."""
        return super().max_e + 1


@dataclass
class NAIProposedFormat(IEEE754BinaryFormat):
    """Used for Nvidia, ARM and Intel's proposed E4M3 format."""

    inf_encoding: ClassVar[str] = "N/A"
    nan_encoding: ClassVar[str] = "E=1s M=1s"

    @property
    def max_e(self) -> int:
        """Maximum exponent. The all 1s exponent no longer denotes NaN/Inf."""
        return super().max_e + 1

    @property
    def abs_max(self) -> float:
        """Absolute maximum representable value.

        Accounts for the fact that the value with all 1s exponent+mantissa denotes NaN.
        This was previously handled by taking one value away from max_e, but now the all
        1s exponent is generally valid, this special-case must be added.
        """
        return float(super().abs_max - 2 ** (self.max_e - self.m_width))


@dataclass
@total_ordering
class FloatInstance:
    """An instance of a floating point number, defined with reference to an instance
    or subclass of `IEEE754BinaryFormat`.
    """

    format: IEEE754BinaryFormat
    s: int
    e: int
    m: int

    def __post_init__(self) -> None:
        assert self.s in [0, 1], self.s
        self.e_limit = int(2**self.format.e_width) - 1
        self.m_limit = int(2**self.format.m_width) - 1
        assert (
            0 <= self.e <= self.e_limit
        ), f"Exponent {self.e} outside range: [0, {self.e_limit}]"
        assert (
            0 <= self.m <= self.m_limit
        ), f"Mantissa {self.m} outside range: [0, {self.m_limit}]"

    @property
    def value(self) -> float:
        """The numerical value of the bitstring, as defined by the supplied format."""
        if self._is_inf():
            return float("inf") * int((-1) ** self.s)
        if self._is_nan():
            return float("nan")
        if self._is_subnormal():
            return self._subnormal_val()
        return self._normal_val()

    def _normal_val(self) -> float:
        e = self.e - self.format.bias
        m = 1 + (self.m / (self.m_limit + 1))
        return float(((-1) ** self.s) * (2**e) * m)

    def _subnormal_val(self) -> float:
        e = 1 - self.format.bias
        m = self.m / (self.m_limit + 1)
        return float(((-1) ** self.s) * (2**e) * m)

    def _is_subnormal(self) -> bool:
        return self.e == 0

    def _is_nan(self) -> bool:
        if self.format.nan_encoding == "E=1s M≠0s":
            return self.e == self.e_limit and self.m != 0
        if self.format.nan_encoding == "E=1s M=1s":
            return self.e == self.e_limit and self.m == self.m_limit
        assert (
            self.format.nan_encoding == "S=1 E=0s M=0s"
        ), f"NaN encoding `'{self.format.nan_encoding}' not recognised"
        return self.s == 1 and self.e == 0 and self.m == 0

    def _is_inf(self) -> bool:
        if self.format.inf_encoding == "E=1s M=0s":
            return self.e == self.e_limit and self.m == 0
        assert (
            self.format.inf_encoding == "N/A"
        ), f"Inf encoding `'{self.format.inf_encoding}' not recognised"
        return False

    def __repr__(self) -> str:
        return str(self.value)

    def __eq__(self, other: Any) -> bool:
        return self.value.__eq__(other)

    def __lt__(self, other: Any) -> bool:
        return self.value.__lt__(other)
