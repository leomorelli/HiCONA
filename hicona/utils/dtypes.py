"""Module with custom data types for the hicona package type annotation."""

from typing import Iterable, Literal, TypeVar

import matplotlib.axes as axes
import pandas as pd


# Type alias for an iterable (usually a generator) of pandas DataFrames.
PdChunks = Iterable[pd.DataFrame]

# Type alias for generic type annotation.
T = TypeVar("T")

# Type alias for the alpha mod type.
AlphaModType = Literal["min", "max"]

# Type alias for the axes type.
OptionalAxes = axes.Axes | None
