"""
Custom cooler file handle to extend cooler.Cooler class functionalities.

HiconaCooler inherits from cooler.Cooler and add some functionalities to it,
mainly the ability to manipulate bin annotations, which are severly limited in
use in the original class.

"""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

import cooler  # type: ignore
import polars as pl

from .._utils.chunked_ops import (
    add_ind_col,
    convert,
    format_stream,
    row_filter,
    cast_dtypes,
)
from .graph import HiconaGraph
from .table import HiconaTable

if TYPE_CHECKING:
    import h5py  # type: ignore
    from .._utils.df_dtypes import DataFrame, DfChunks, PlChunks, Bool, DfDtype


__all__ = ["HiconaCooler"]


def _chunked_selector(selector, chunk_size) -> "PlChunks":
    """Take a RangeSelector1D from Cooler and return it in chunks."""

    num_chunks: int = ceil(len(selector) / chunk_size)
    for i in range(num_chunks):
        start: int = i * chunk_size
        stop: int = min((i + 1) * chunk_size, len(selector))
        yield pl.from_pandas(selector[start:stop])


class HiconaCooler(cooler.Cooler):
    """Cooler file handle with extended functionalities."""

    def __init__(self, store: str | "h5py.File" | "h5py.Group", **kwargs):
        # Mask deprecated root parameter from super-class
        super().__init__(store, **kwargs)

    def get_pixels(
        self,
        region: str | None = None,
        *,
        df_dtype: DfDtype = "polars",
        as_chunks: Bool = True,
        chunk_size: int = 10_000_000,
    ) -> "DataFrame" | "DfChunks":
        """Return an iterable of pixel chunks."""

        # NOTE: Using this implementation rather than the one in pixel_ops.py since
        #       it is more efficient due to having access to the extent method.

        chunks: "PlChunks" = _chunked_selector(self.pixels(), chunk_size)

        if region:
            lower, upper = self.extent(region)
            bin1_bounds = (pl.col("bin1_id") >= lower) & (pl.col("bin1_id") < upper)
            bin2_bounds = (pl.col("bin2_id") >= lower) & (pl.col("bin2_id") < upper)
            expr = bin1_bounds & bin2_bounds

            chunks = row_filter(convert(chunks, "polars"), expr, chunk_size)

        return format_stream(chunks, df_dtype, as_chunks)

    def get_bins(
        self,
        region: str | None = None,
        *,
        df_dtype: DfDtype = "polars",
        as_chunks: Bool = True,
        chunk_size: int = 10_000_000,
    ) -> "DataFrame" | "DfChunks":
        """Return an iterable of bin chunks.

        Notes
        -----
        For small bin tables, which is often the case, this method is algorithmically
        more expensive than using the ``bins()`` method from the cooler.Cooler class.
        The advantage of this method is that it allows to read in chunks large bin
        tables, which could be the case for very high resolution Hi-C data. Moreover,
        for small bin tables, the difference in performance should be negligible but
        with the advantage of being able to return the bins as polars.DataFrame objects.

        """

        chunks: "PlChunks" = _chunked_selector(self.bins(), chunk_size)
        chunks = add_ind_col(chunks, "bin_id")

        if region:
            lower, upper = self.extent(region)
            expr = (pl.col("bin_id") >= lower) & (pl.col("bin_id") < upper)
            chunks = row_filter(chunks, expr, chunk_size)

        chunks = cast_dtypes(chunks, {"chrom": str})
        return format_stream(chunks, df_dtype, as_chunks)

    def bin_annot_add(self):
        """Add a new bin annotation column to the bin table in the file."""

    def bin_annot_del(self):
        """Delete a bin annotation column from the bin table in the file."""

    def bin_annot_list(self):
        """Return an iterable of the bin annotation columns in the file."""

    def get_graph(self, region: str | None) -> "HiconaGraph":
        """Create and return an instance of HiconaGraph."""
        return HiconaGraph.from_cooler(self, region=region)

    def get_table(
        self, region: str | None, *, store_size: int = 10_000_000
    ) -> "HiconaTable":
        """Create and return an instance of HiconaTable."""
        return HiconaTable.from_cooler(self, region=region, store_size=store_size)
