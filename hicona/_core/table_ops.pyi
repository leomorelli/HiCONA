from typing import Literal, overload

import pandas as pd
import polars as pl

from .._utils.df_dtypes import DataFrame, DfStream, PdChunks, PlChunks

# subset_pixel_region
@overload
def subset_pixel_region(
    pixels: DataFrame | DfStream = ...,
    *,
    region: str = ...,
    ref_bins: DataFrame | DfStream = ...,
    df_dtype: Literal["pandas"],
    as_chunks: Literal[True],
    chunk_size: int = ...,
) -> PdChunks: ...
@overload
def subset_pixel_region(
    pixels: DataFrame | DfStream = ...,
    *,
    region: str = ...,
    ref_bins: DataFrame | DfStream = ...,
    df_dtype: Literal["polars"],
    as_chunks: Literal[True],
    chunk_size: int = ...,
) -> PlChunks: ...
@overload
def subset_pixel_region(
    pixels: DataFrame | DfStream = ...,
    *,
    region: str = ...,
    ref_bins: DataFrame | DfStream = ...,
    df_dtype: Literal["pandas"],
    as_chunks: Literal[False],
    chunk_size: int = ...,
) -> pd.DataFrame: ...
@overload
def subset_pixel_region(
    pixels: DataFrame | DfStream = ...,
    *,
    region: str = ...,
    ref_bins: DataFrame | DfStream = ...,
    df_dtype: Literal["polars"],
    as_chunks: Literal[False],
    chunk_size: int = ...,
) -> pl.DataFrame: ...

# subset_bin_region
@overload
def subset_bin_region(
    bins: DataFrame | DfStream = ...,
    *,
    region: str = ...,
    df_dtype: Literal["pandas"],
    as_chunks: Literal[True],
    chunk_size: int = ...,
) -> PdChunks: ...
@overload
def subset_bin_region(
    bins: DataFrame | DfStream = ...,
    *,
    region: str = ...,
    df_dtype: Literal["polars"],
    as_chunks: Literal[True],
    chunk_size: int = ...,
) -> PlChunks: ...
@overload
def subset_bin_region(
    bins: DataFrame | DfStream = ...,
    *,
    region: str = ...,
    df_dtype: Literal["pandas"],
    as_chunks: Literal[False],
    chunk_size: int = ...,
) -> pd.DataFrame: ...
@overload
def subset_bin_region(
    bins: DataFrame | DfStream = ...,
    *,
    region: str = ...,
    df_dtype: Literal["polars"],
    as_chunks: Literal[False],
    chunk_size: int = ...,
) -> pl.DataFrame: ...
