"""Palettes and colors used throughout HiCONA"""

from enum import Enum

import seaborn as sns

WHITE_HEX: str = "#FFFFFF"

CLUSTER = sns.color_palette("tab20", as_cmap=True)
CLUSTER.set_under("whitesmoke")


class Palettes(Enum):
    """All palettes used throughout the HiCONA."""

    COUNTS = sns.color_palette("YlOrRd", as_cmap=True)
    PROB = sns.color_palette("coolwarm", as_cmap=True)
    CLUST = CLUSTER
