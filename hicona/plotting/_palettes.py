"""Palettes and colors used throughout HiCONA"""

from enum import Enum

import seaborn as sns
import matplotlib.colors as co


WHITE_HEX: str = "#FFFFFF"


class Palettes(Enum):
    """All palettes used throughout the HiCONA."""

    PROB = sns.color_palette("coolwarm", as_cmap=True)
    CLUST = "tab20"
    WHITE_ZERO_CLUST = co.ListedColormap(
        [WHITE_HEX] + [co.to_hex(c) for c in sns.color_palette("tab20", 500)],
    )
