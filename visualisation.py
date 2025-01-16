"""Generate visalisations of representable numbers in various formats.
"""
from typing import Dict, Union

import altair as alt
import numpy as np
import pandas as pd
from scipy.stats import norm

from float_format import IEEE754BinaryFormat
from histogram import log_histogram
from int_format import ScaledTwosComplementFormat


def table_viz(
    formats: Dict[str, Union[IEEE754BinaryFormat, ScaledTwosComplementFormat]],
    file_name="out/table.csv",
) -> None:
    """Generate a csv table describing representable values in different formats.

    Args:
        formats (Dict[str, Union[IEEE754BinaryFormat, ScaledTwosComplementFormat]]):
            dict of formats to visualise, of type: {"format name": FormatClass}.
        file_name (str, optional): out file. Defaults to "out/table.csv".
    """
    df = pd.DataFrame(
        {
            "IEEE compliant": "",
            "E bits": "",
            "M bits": "",
            "Max": "2^max_e * (2 - 2^-M)",
            "|Min normal|": "2^min_e",
            "|Min|": "2^(min_e - M)",
            "Bias": "2^(E-1) - 1",
            "Max exp": "2^(E-1) - 1",
            "Min exp": "2 - 2^(E-1)",
            "Inf encoding": IEEE754BinaryFormat.inf_encoding,
            "NaN encoding": IEEE754BinaryFormat.nan_encoding,
            "Zero encoding": IEEE754BinaryFormat.zero_encoding,
        },
        index=["IEEE Standard"],
    )

    for name, fmt in formats.items():
        max_value = (
            f"{float(fmt.abs_max):.2}" if fmt.abs_max > 10e5 else int(fmt.abs_max)
        )
        abs_min_value = fmt.abs_min if fmt.abs_min >= 1 else f"{float(fmt.abs_min):.2}"
        d = {
            "IEEE compliant": type(fmt) is IEEE754BinaryFormat,
            "Max": max_value,
            "|min|": abs_min_value,
        }
        if issubclass(type(fmt), IEEE754BinaryFormat):
            d.update(
                {
                    "E bits": fmt.e_width,
                    "M bits": fmt.m_width,
                    "|Min normal|": f"{float(fmt.abs_min_normal):.2}",
                    "Bias": fmt.bias,
                    "Max exp": fmt.max_e,
                    "Min exp": fmt.min_e,
                    "Inf encoding": fmt.inf_encoding,
                    "NaN encoding": fmt.nan_encoding,
                    "Zero encoding": fmt.zero_encoding,
                }
            )
        else:
            for c in df.columns:
                if c not in d:
                    d[c] = ""
        df = pd.concat([df, pd.DataFrame(d, index=[name])])

    csv = df.to_csv()
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(csv)


def histogram_viz(
    formats: Dict[str, Union[IEEE754BinaryFormat, ScaledTwosComplementFormat]],
    hist_bin_width: float,
    sample_count: int,
    exclude_fp32: bool = False,  # gives speedup
    file_name: str = "out/histogram.html",
) -> alt.Chart:
    """Generate an interactive histogram.

    Args:
        formats (Dict[str, Union[IEEE754BinaryFormat, ScaledTwosComplementFormat]]):
            dict of formats to visualise, of type: {"format name": FormatClass}.
        hist_bin_width (float): width of histogram bins (log value).
        sample_count (int): number of uniform samples to take of the values in each
            format (there are too many FP32 values to loop through)
        exclude_fp32 (bool, optional): used to speed up results. Defaults to False.

    Returns:
        _type_: interactive chart.
    """
    float_chart_data = pd.DataFrame()
    for format_idx, (format_name, format) in enumerate(formats.items()):
        if exclude_fp32 and format_name == "FP32":
            continue
        hist_df = log_histogram(format, hist_bin_width, sample_count)
        hist_df["number_format"] = format_name
        hist_df["format_idx"] = format_idx
        float_chart_data = pd.concat([float_chart_data, hist_df])

    start = -18
    end = 3
    bin_edges = np.arange(start, end, hist_bin_width)
    bin_edges = np.append(bin_edges, end)
    bin_cdfs = norm.cdf(np.power(2, bin_edges))
    bin_probs = np.ediff1d(bin_cdfs)
    with np.errstate(divide="ignore"):
        count = np.log2(bin_probs)

    normal_name = "normal dist. (µ=0, σ^2)"
    normal_hist_df = pd.DataFrame(
        {
            "number_format": normal_name,
            "start": bin_edges[:-1],
            "end": bin_edges[1:],
            "count": count,
            "format_idx": len(formats),
        }
    )

    chart_data = pd.concat([float_chart_data, normal_hist_df])
    chart_data["count"] += 1  # hack to make y-axis log scale work

    color_list = [
        "#9ecae9",
        "#a6d27a",
        "#ffe155",
        "#ffbbad",
        "#9d755d",
        "#e07104",
        "#e0a513",
        "#b279a2",
        "#ff9da6",
        "#e45756",
        "#6fa0d6",
    ]
    legend_keys = list(formats.keys())
    color = alt.Color(
        "number_format:N",
        legend=None,
        sort=legend_keys,
        scale=alt.Scale(range=color_list),
    )

    normal_std_slider = alt.binding_range(min=-140, max=140, step=1, name="log₂(normal_σ):")
    normal_samples_slider = alt.binding_range(
        min=0, max=20, step=1, name="log₂(normal_samples):"
    )
    normal_std_selector = alt.selection_single(
        name="normal_std_selector",
        fields=["log_std"],
        bind=normal_std_slider,
        value=[{"log_std": 0}],
    )
    normal_samples_selector = alt.selection_single(
        name="normal_samples_selector",
        fields=["log_normal_samples"],
        bind=normal_samples_slider,
        value=[{"log_normal_samples": 12}],
    )
    distr_selector = alt.selection_multi(
        fields=["number_format"],
        value=[{"number_format": format_name} for format_name in formats.keys()]
        + [{"number_format": normal_name}],
    )

    main_chart = (
        alt.Chart(chart_data)
        .mark_bar(
            opacity=0.8,
            binSpacing=0,
        )
        .transform_calculate(
            xstart=f"datum.start + (datum.number_format == '{normal_name}' ?"
            " normal_std_selector.log_std * 1.0 : 0.0)",
            xend=f"datum.end + (datum.number_format == '{normal_name}' ?"
            " normal_std_selector.log_std * 1.0 : 0.0)",
            count_adj=f"datum.number_format == '{normal_name}' ?"
            " max((datum.count + normal_samples_selector.log_normal_samples * 1.0), 0) :"
            " datum.count",
        )
        .encode(
            alt.X(
                "xstart:Q",
                title="representable values, log₂",
                bin=alt.Bin(binned=True),
            ),
            alt.X2("xend", title="representable values, log₂"),
            alt.Y(
                "count_adj:Q",
                title="bin count",
                axis=alt.Axis(
                    labelExpr="if(datum.value < 1, '', pow(2, datum.value-1))",
                    tickMinStep=1,
                ),
            ),
            color,
            alt.Order("format_idx:N"),
        )
        .add_selection(normal_samples_selector)
        .add_selection(normal_std_selector)
        .transform_filter(distr_selector)
        .properties(
            width=550,
            height=400,
        )
        .interactive(bind_y=False)
    )

    legend = (
        alt.Chart(chart_data)
        .mark_rect()
        .encode(
            alt.Y(
                "number_format:N",
                axis=alt.Axis(orient="right"),
                sort=legend_keys,
                title="number format",
            ),
            color=alt.condition(distr_selector, color, alt.value("lightgray")),
        )
        .properties(
            width=25,
            height=250,
        )
        .add_selection(distr_selector)
    )

    total_chart = (main_chart | legend).configure(
        background='#FCFCFC'
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14,
    )
    total_chart.save(file_name)
    return total_chart
