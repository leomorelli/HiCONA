from typing import Any, Literal, overload

import pandas as pd
import polars as pl

from .._utils.df_dtypes import DataFrame, DfStream, PdChunks, PlChunks
from .cooler import HiconaCooler

class HiconaTable:
    """Class to handle data subsets from a cooler file."""

    def __init__(
        self,
        *,
        bins: DataFrame | DfStream = ...,
        pixels: DataFrame | DfStream = ...,
        info: dict[str, Any] = ...,
        store_size: int = ...,
    ): ...

    # get_pixels
    @overload
    def get_pixels(
        self,
        region: str | None = ...,
        *,
        df_dtype: Literal["pandas"],
        as_chunks: Literal[True],
        chunk_size: int = ...,
    ) -> PdChunks: ...
    @overload
    def get_pixels(
        self,
        region: str | None = ...,
        *,
        df_dtype: Literal["polars"],
        as_chunks: Literal[True],
        chunk_size: int = ...,
    ) -> PlChunks: ...
    @overload
    def get_pixels(
        self,
        region: str | None = ...,
        *,
        df_dtype: Literal["pandas"],
        as_chunks: Literal[False],
        chunk_size: int = ...,
    ) -> pd.DataFrame: ...
    @overload
    def get_pixels(
        self,
        region: str | None = ...,
        *,
        df_dtype: Literal["polars"],
        as_chunks: Literal[False],
        chunk_size: int = ...,
    ) -> pl.DataFrame: ...

    # get_bins
    @overload
    def get_bins(
        self,
        region: str | None = None,
        *,
        df_dtype: Literal["pandas"],
        as_chunks: Literal[True],
        chunk_size: int = ...,
    ) -> PdChunks: ...
    @overload
    def get_bins(
        self,
        region: str | None = None,
        *,
        df_dtype: Literal["polars"],
        as_chunks: Literal[True],
        chunk_size: int = ...,
    ) -> PlChunks: ...
    @overload
    def get_bins(
        self,
        region: str | None = None,
        *,
        df_dtype: Literal["pandas"],
        as_chunks: Literal[False],
        chunk_size: int = ...,
    ) -> pd.DataFrame: ...
    @overload
    def get_bins(
        self,
        region: str | None = None,
        *,
        df_dtype: Literal["polars"],
        as_chunks: Literal[False],
        chunk_size: int = ...,
    ) -> pl.DataFrame: ...

    # info
    @property
    def info(self) -> dict[str, Any]: ...

    # from_cooler
    @classmethod
    def from_cooler(
        cls,
        handle: HiconaCooler = ...,
        *,
        region: str | None = ...,
        store_size: int = ...,
    ) -> "HiconaTable": ...
