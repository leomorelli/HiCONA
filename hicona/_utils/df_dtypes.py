"""Custom data types for type hinting."""

from collections.abc import Generator, Iterator, Iterable
from typing import Literal

import pandas as pd
import polars as pl

DataFrame = pd.DataFrame | pl.DataFrame

PlChunks = Generator[pl.DataFrame, None, None]
PdChunks = Generator[pd.DataFrame, None, None]
DfChunks = PlChunks | PdChunks

PlStream = PlChunks | Iterable[pl.DataFrame] | Iterator[pl.DataFrame]
PdStream = PdChunks | Iterable[pd.DataFrame] | Iterator[pd.DataFrame]
DfStream = PlStream | PdStream

Bool = Literal[True] | Literal[False]
DfDtype = Literal["pandas"] | Literal["polars"]

# PlCollection = Iterable[pl.DataFrame] | Iterator[pl.DataFrame]
# PdCollection = Iterable[pd.DataFrame] | Iterator[pd.DataFrame]
# DfCollection = PlCollection | PdCollection

# DfOrCollection = DataFrame | DfCollection
