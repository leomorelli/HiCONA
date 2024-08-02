"""Module with custom data types for the hicona package type annotation."""

from typing import Any, Iterable, Literal, TypeVar, Union, TYPE_CHECKING

import matplotlib.axes as ax
import polars as pl
import pandas as pd

if TYPE_CHECKING:
    from hicona.preprocess import FiltOperation, NormOperation, SparOperation

# Type alias for an iterable (usually a generator) of pandas DataFrames.
DfChunks = Iterable[pl.DataFrame]

# Type alias for generic type annotation.
T = TypeVar("T")

# Type alias for the alpha mod type.
AlphaModType = Literal["alpha_min", "alpha_max"]

# Type alias for the axes type.
OptionalAxes = ax.Axes | None

# Type alias for a kwargs-like dict.
KwargsDict = dict[str, Any]

# Type alias for the JSON dictionary.
JsonDict = dict[str, dict[str, Any]]


Operation = Union["FiltOperation", "NormOperation", "SparOperation"]

GenericDf = Union[pl.DataFrame, pd.DataFrame]
