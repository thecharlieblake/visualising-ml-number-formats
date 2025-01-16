# %%
"""Generate an html table and histogram to visualise floating point formats.
"""
from typing import Dict, Union

from float_format import GAQProposedFormat, IEEE754BinaryFormat, NAIProposedFormat
from int_format import ScaledTwosComplementFormat, TwosComplementFormat
from visualisation import histogram_viz, table_viz

# %reload_ext autoreload
# %autoreload 2
#%%
if __name__ == "__main__":
    # %%
    formats: Dict[str, Union[IEEE754BinaryFormat, TwosComplementFormat]] = {
        "FP32": IEEE754BinaryFormat(8, 23),
        "TF32": IEEE754BinaryFormat(8, 10),
        "BF16": IEEE754BinaryFormat(8, 7),
        "FP16": IEEE754BinaryFormat(5, 10),
        # "FP8_1.5.2_GAQ": GAQProposedFormat(5, 2, custom_bias=16),
        "FP8_E5": IEEE754BinaryFormat(5, 2),
        "INT8x512": ScaledTwosComplementFormat(8, scale=2**9),
        # "FP8_1.4.3_GAQ": GAQProposedFormat(4, 3, custom_bias=8),
        "FP8_E4": NAIProposedFormat(4, 3),
        "INT8x2": ScaledTwosComplementFormat(8, scale=2),
    }
    table_viz(formats)
    histogram_viz(
        formats,
        hist_bin_width=2**-2,
        sample_count=2**15,
        # exclude_fp32=True,  # TODO: change for git
    )

# %%
