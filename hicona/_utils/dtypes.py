"""Module with custom data types for the hicona package type annotation."""

from typing import Any, Iterable, Literal, TypeVar

from matplotlib import axes
import pandas as pd


# Type alias for an iterable (usually a generator) of pandas DataFrames.
PdChunks = Iterable[pd.DataFrame]

# Type alias for generic type annotation.
T = TypeVar("T")

# Type alias for the alpha mod type.
AlphaModType = Literal["alpha_min", "alpha_max"]

# Type alias for the axes type.
OptionalAxes = axes.Axes | None

# Type alias for the options dictionary.
OptionsDict = dict[str, Any]
