from typing import Any, Literal, overload

import pandas as pd
import polars as pl

from .._utils.df_dtypes import PdChunks, PlChunks, Bool
from .cooler import HiconaCooler
from .table import HiconaTable

class HiconaGraph:

    # get_bins
    @overload
    def get_bins(
        self,
        region: str | None = ...,
        *,
        bare: Bool = ...,
        df_dtype: Literal["pandas"],
        as_chunks: Literal[True],
        chunk_size: int = ...,
    ) -> PdChunks: ...
    @overload
    def get_bins(
        self,
        region: str | None = ...,
        *,
        bare: Bool = ...,
        df_dtype: Literal["polars"],
        as_chunks: Literal[True],
        chunk_size: int = ...,
    ) -> PlChunks: ...
    @overload
    def get_bins(
        self,
        region: str | None = ...,
        *,
        bare: Bool = ...,
        df_dtype: Literal["pandas"],
        as_chunks: Literal[False],
        chunk_size: int = ...,
    ) -> pd.DataFrame: ...
    @overload
    def get_bins(
        self,
        region: str | None = ...,
        *,
        bare: Bool = ...,
        df_dtype: Literal["polars"],
        as_chunks: Literal[False],
        chunk_size: int = ...,
    ) -> pl.DataFrame: ...

    # get_pixels
    @overload
    def get_pixels(
        self,
        region: str | None = ...,
        *,
        keep_genomic: Bool = ...,
        df_dtype: Literal["pandas"],
        as_chunks: Literal[True],
        chunk_size: int = ...,
    ) -> PdChunks: ...
    @overload
    def get_pixels(
        self,
        region: str | None = ...,
        *,
        keep_genomic: Bool = ...,
        df_dtype: Literal["polars"],
        as_chunks: Literal[True],
        chunk_size: int = ...,
    ) -> PlChunks: ...
    @overload
    def get_pixels(
        self,
        region: str | None = ...,
        *,
        keep_genomic: Bool = ...,
        df_dtype: Literal["pandas"],
        as_chunks: Literal[False],
        chunk_size: int = ...,
    ) -> pd.DataFrame: ...
    @overload
    def get_pixels(
        self,
        region: str | None = ...,
        *,
        keep_genomic: Bool = ...,
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
        handle: "HiconaCooler",
        *,
        region: str | None = None,
    ) -> "HiconaGraph": ...

    # from_table
    @classmethod
    def from_table(
        cls,
        table: "HiconaTable",
        *,
        region: str | None = None,
    ) -> "HiconaGraph": ...
