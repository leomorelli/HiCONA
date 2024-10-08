from typing import Literal, overload

import cooler  # type: ignore
import pandas as pd
import polars as pl

from .._utils.df_dtypes import PdChunks, PlChunks
from .graph import HiconaGraph
from .table import HiconaTable

class HiconaCooler(cooler.Cooler):

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
        region: str | None = ...,
        *,
        df_dtype: Literal["pandas"],
        as_chunks: Literal[True],
        chunk_size: int = ...,
    ) -> PdChunks: ...
    @overload
    def get_bins(
        self,
        region: str | None = ...,
        *,
        df_dtype: Literal["polars"],
        as_chunks: Literal[True],
        chunk_size: int = ...,
    ) -> PlChunks: ...
    @overload
    def get_bins(
        self,
        region: str | None = ...,
        *,
        df_dtype: Literal["pandas"],
        as_chunks: Literal[False],
        chunk_size: int = ...,
    ) -> pd.DataFrame: ...
    @overload
    def get_bins(
        self,
        region: str | None = ...,
        *,
        df_dtype: Literal["polars"],
        as_chunks: Literal[False],
        chunk_size: int = ...,
    ) -> pl.DataFrame: ...

    # get_table
    def get_table(
        self, region: str | None = ..., *, store_size: int = ...
    ) -> "HiconaTable": ...

    # get_graph
    def get_graph(self, region: str | None = ...) -> "HiconaGraph": ...
