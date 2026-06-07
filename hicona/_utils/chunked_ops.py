"""
Functions to rechunk an iterable of pl.DataFrame objects.

Applying operations to a DataFrame changes the size of the DataFrame,
therefore the iterator is rechunked to keep the DataFrame size constant.
This is especially useful if many DataFrames become empty, thus avoiding
unnecessary merges.
"""

from __future__ import annotations

from typing import Literal, overload

import pandas as pd
import polars as pl

from .df_dtypes import (
    Bool,
    DataFrame,
    DfChunks,
    DfDtype,
    DfStream,
    PdChunks,
    PlChunks,
    PlStream,
)


@overload
def convert(chunks: DfStream, to: Literal["polars"]) -> PlChunks: ...


@overload
def convert(chunks: DfStream, to: Literal["pandas"]) -> PdChunks: ...


def convert(chunks: DfStream, to: DfDtype) -> DfChunks:
    """Convert a stream of DataFrames to the specified dtype."""

    if to == "polars":
        return (c if isinstance(c, pl.DataFrame) else pl.from_pandas(c) for c in chunks)
    if to == "pandas":
        return (c if isinstance(c, pd.DataFrame) else c.to_pandas() for c in chunks)
    raise ValueError("Invalid conversion target.")


@overload
def format_stream(
    stream: DfStream,
    to: Literal["polars"],
    as_chunks: Literal[True],
) -> PlChunks: ...


@overload
def format_stream(
    stream: DfStream,
    to: Literal["polars"],
    as_chunks: Literal[False],
) -> pl.DataFrame: ...


@overload
def format_stream(
    stream: DfStream,
    to: Literal["pandas"],
    as_chunks: Literal[True],
) -> PdChunks: ...


@overload
def format_stream(
    stream: DfStream,
    to: Literal["pandas"],
    as_chunks: Literal[False],
) -> pd.DataFrame: ...


def format_stream(
    stream: DfStream,
    to: DfDtype,
    as_chunks: Bool,
) -> DataFrame | DfChunks:
    """Convert an iterable of DataFrames to a DataFrame or an iterable of DataFrames."""

    match to, as_chunks:
        case "polars", True:
            return convert(stream, to)
        case "pandas", True:
            return convert(stream, to)
        case "polars", False:
            return pl.concat(convert(stream, to))
        case "pandas", False:
            return pd.concat(convert(stream, to))
        case _:
            raise ValueError("Invalid conversion target.")


def rechunk(chunks: PlStream, size: int) -> PlChunks:
    """Return an iterable of DataFrames with a constant size."""

    total_size: int = 0
    parts: list[pl.DataFrame] = []

    for chunk in chunks:
        total_size += chunk.height
        parts.append(chunk)

        if total_size >= size:
            data = pl.concat(parts)

            while data.height >= size:
                yield data.slice(0, size)
                data = data.slice(size)

            parts = [data]
            total_size = data.height

    if parts:
        data = pl.concat(parts)
        if data.height > 0:
            yield data


def row_filter(chunks: PlStream, expr: pl.Expr, size: int | None = None) -> PlChunks:
    """Filter rows of a DataFrame using an expression and rechunk the result."""

    def filt_chunks(chunks: PlStream, expr: pl.Expr) -> PlChunks:
        for chunk in chunks:
            yield chunk.filter(expr)

    if size:
        return rechunk(filt_chunks(chunks, expr), size)
    return filt_chunks(chunks, expr)


def add_ind_col(chunks: PlStream, ind_name: str, offset: int = 0) -> PlChunks:
    """Analogous to adding an index column in polars but works in chunks."""

    curr_offset: int = offset
    stream_has_ind: bool | None = None

    for chunk in chunks:
        chunk_has_ind: bool = ind_name in chunk.columns
        stream_has_ind = stream_has_ind or chunk_has_ind

        if stream_has_ind != chunk_has_ind:
            raise ValueError("All chunks should have an index column or none of them.")

        if chunk_has_ind:
            yield chunk
        else:
            yield chunk.with_row_index(ind_name, curr_offset).with_columns(
                pl.col(ind_name).cast(pl.Int64)
            )

        curr_offset += chunk.height


def to_iterable(*args: DataFrame | DfStream) -> tuple[DfStream, ...]:
    """Convert a variable number of arguments to iterables."""

    def to_iter(arg: DataFrame | DfStream) -> DfStream:
        # NOTE: Type checks must be split, else mypy will think that [arg] is
        #       Iterable[pl.DataFrame | pd.DataFrame] rather than
        #       Iterable[pl.DataFrame] | Iterable[pd.DataFrame].
        arg = [arg] if isinstance(arg, pl.DataFrame) else arg
        arg = [arg] if isinstance(arg, pd.DataFrame) else arg
        return arg

    return tuple(map(to_iter, args))


def cast_dtypes(chunks: PlStream, dtypes: dict) -> PlChunks:
    """Cast the columns of a DataFrame to the specified types."""

    # TODO: find a way to type hint the input dict
    for chunk in chunks:
        yield chunk.cast(dtypes)
