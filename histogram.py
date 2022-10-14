"""Given a number format, analyse its distribution of values.
"""
import math
from typing import Tuple, Union

import numpy as np
import pandas as pd

from float_format import FloatInstance, IEEE754BinaryFormat
from int_format import ScaledTwosComplementFormat, TwosComplementInstance


def _is_valid_number(x: float) -> bool:
    return not math.isnan(x) and x not in [float("-inf"), float("inf")]


def _log_pos_floats(
    fmt: IEEE754BinaryFormat, sample_count: int = 2**13
) -> np.ndarray:

    mantissa_samples = sample_count // (2**fmt.e_width)
    step_size = max((2**fmt.m_width) // mantissa_samples, 1)

    normal_values = [
        FloatInstance(fmt, 0, e, m).value
        for e in range(1, 2**fmt.e_width)
        for m in range(0, 2**fmt.m_width, step_size)
    ]
    subnormal_values = [
        FloatInstance(fmt, 0, 0, m).value
        for m in range(1, 2**fmt.m_width)  # Don't sample subnormals
    ]
    values = subnormal_values + normal_values
    return np.log2([v for v in values if _is_valid_number(v)])


def _log_pos_ints(fmt: ScaledTwosComplementFormat) -> np.ndarray:
    return np.log2(
        [
            TwosComplementInstance(fmt, uint).value * fmt.scale
            for uint in range(1, 2 ** (fmt.width - 1))
        ]
    )


def _count_float_bins(
    fmt: IEEE754BinaryFormat, bin_edges: np.ndarray, sample_count: int
) -> Tuple[np.ndarray, np.ndarray]:
    floats = _log_pos_floats(fmt, sample_count)
    hist_counts, bin_edges = np.histogram(floats, bin_edges)

    sampling_factor = max(2 ** (fmt.e_width + fmt.m_width) // sample_count, 1)
    hist_counts = np.where(
        bin_edges[:-1] < np.log2(fmt.abs_min_normal),
        hist_counts,
        hist_counts * sampling_factor,
    )
    return hist_counts, bin_edges


def _count_int_bins(fmt: ScaledTwosComplementFormat, bin_edges: np.ndarray):
    ints = _log_pos_ints(fmt)
    return np.histogram(ints, bin_edges)


def log_histogram(
    fmt: Union[IEEE754BinaryFormat, ScaledTwosComplementFormat],
    bin_width: float = 2**-2,
    sample_count: int = 2**15,
) -> pd.DataFrame:
    """Return a histogram of the values expressible in the given format, where both the
    x and y-axes are on a log-scale.

    Args:
        fmt (Union[IEEE754BinaryFormat, ScaledTwosComplementFormat]): the format to
            generate a histogram for
        bin_width (float, optional): width of each bin in log space. Defaults to 2**-2.
        sample_count (int, optional): enumerating all values of wider formats is
            computationally expensive. This defines the number of uniform samples taken
            over the values defined by the format, to be used in the histogram.
            Defaults to 2**15.

    Returns:
        pd.DataFrame: the histogram. Contains the counts, bin starts, and bin ends.
    """
    min_log_value = int(np.ceil(np.log2(fmt.abs_min)))
    max_log_value = int(np.ceil(np.log2(fmt.abs_max)))
    bin_edges = np.arange(min_log_value, max_log_value + 1, bin_width)

    if isinstance(fmt, IEEE754BinaryFormat):
        counts, bin_edges = _count_float_bins(fmt, bin_edges, sample_count)
    else:
        assert isinstance(fmt, ScaledTwosComplementFormat)
        counts, bin_edges = _count_int_bins(fmt, bin_edges)
    with np.errstate(divide="ignore"):
        counts = np.log2(counts)

    return pd.DataFrame(
        {"start": bin_edges[:-1], "end": bin_edges[1:], "count": counts}
    )
