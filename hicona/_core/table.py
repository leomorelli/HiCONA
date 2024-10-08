"""
Class to handle data subsets from a cooler file which are temporarily saved to disk.

HiconaTable creates two temporary storages, one for the bins and one for the pixels.
These storages are used to store the data from a specific region of the cooler file
without having to iterate through (and decompress) the whole file every time.
Temporary storages are torn down when the instance is deleted.

"""

from __future__ import annotations

import copy
import os
from typing import Any, TYPE_CHECKING

import polars as pl

from .._utils.chunked_ops import (
    rechunk,
    convert,
    to_iterable,
    format_stream,
    cast_dtypes,
)
from .._utils.tmp_storage import TmpStorage
from .table_ops import subset_pixel_region, subset_bin_region

if TYPE_CHECKING:
    from .._utils.df_dtypes import (
        DataFrame,
        DfChunks,
        DfStream,
        PlChunks,
        PlStream,
        DfDtype,
        Bool,
    )
    from .cooler import HiconaCooler
    from .graph import HiconaGraph

__all__ = ["HiconaTable"]


class TmpTable(TmpStorage):
    """Parquet table saved in a temporary storage."""

    def __init__(self, prefix: str, chunk_size: int = 10_000_000):
        super().__init__(prefix)
        self._chunk_size = chunk_size

    def dataframe(self) -> pl.LazyFrame:
        """Return the table as a dataframe."""
        return pl.scan_parquet(self.tmp_store)

    def chunks(self) -> "PlChunks":
        """Return the table as a generator of chunks."""

        files = os.listdir(self.tmp_store)
        files.sort()

        for file in files:
            yield pl.read_parquet(os.path.join(self.tmp_store, file))

    def _save_chunks(self, chunks: "PlStream"):
        """Save the provided chunks into the tmp folder."""

        for i, chunk in enumerate(rechunk(chunks, self._chunk_size)):
            chunk.write_parquet(
                self.tmp_store / f"chunk_{str(i).zfill(4)}.parquet",
                statistics=False,
            )


class PixelTable(TmpTable):
    """Parquet table for pixel data saved in a temporary storage."""

    def __init__(self, pixels: "PlStream", store_size: int = 10_000_000):
        super().__init__("hicona_pixels", store_size)
        self._save_chunks(convert(pixels, "polars"))


class BinTable(TmpTable):
    """Parquet table for bin data saved in a temporary storage."""

    def __init__(self, bins: "PlStream", store_size: int = 10_000_000):
        super().__init__("hicona_bins", store_size)
        self._save_chunks(convert(bins, "polars"))


class HiconaTable:
    """Class to handle data subsets from a cooler file."""

    def __init__(
        self,
        *,
        bins: "DataFrame" | "DfStream",
        pixels: "DataFrame" | "DfStream",
        info: dict[str, Any],
        store_size: int = 10_000_000,
    ):

        bins, pixels = to_iterable(bins, pixels)

        self._bins: BinTable = BinTable(convert(bins, "polars"), store_size)
        self._pixels: PixelTable = PixelTable(convert(pixels, "polars"), store_size)
        self._info: dict[str, Any] = info

    def get_pixels(
        self,
        region: str | None = None,
        *,
        df_dtype: DfDtype = "polars",
        as_chunks: Bool = True,
        chunk_size: int = 10_000_000,
    ) -> "DataFrame" | "DfChunks":
        """Return an iterable of pixel chunks."""

        chunks: "PlChunks" = self._pixels.chunks()

        if region:
            chunks = subset_pixel_region(
                chunks,
                region=region,
                ref_bins=self._bins.chunks(),
                df_dtype="polars",
                as_chunks=True,
                chunk_size=chunk_size,
            )

        chunks = rechunk(chunks, chunk_size)
        return format_stream(chunks, df_dtype, as_chunks)

    def get_bins(
        self,
        region: str | None = None,
        *,
        df_dtype: DfDtype = "polars",
        as_chunks: Bool = True,
        chunk_size: int = 10_000_000,
    ) -> "DataFrame" | "DfChunks":
        """Return the bins as a dataframe."""

        chunks: "PlChunks" = self._bins.chunks()

        if region:
            chunks = subset_bin_region(
                chunks,
                region=region,
                df_dtype="polars",
                as_chunks=True,
                chunk_size=chunk_size,
            )

        chunks = rechunk(chunks, chunk_size)
        chunks = cast_dtypes(chunks, {"chrom": str})
        return format_stream(chunks, df_dtype, as_chunks)

    @property
    def info(self) -> dict[str, Any]:
        """Dictionary with additional information about the table."""
        return copy.deepcopy(self._info)

    @classmethod
    def from_graph(cls, graph: "HiconaGraph") -> "HiconaTable":
        """Create a HiconaTable from a HiconaGraph instance."""

        pixels: "PlChunks" = graph.get_pixels(df_dtype="polars", as_chunks=True)
        bins: "PlChunks" = graph.get_bins(df_dtype="polars", as_chunks=True)
        info = graph.info

        return cls(bins=bins, pixels=pixels, info=info)

    @classmethod
    def from_cooler(
        cls,
        handle: "HiconaCooler",
        *,
        region: str | None,
        store_size: int = 10_000_000,
    ) -> "HiconaTable":
        """Create a HiconaTable from a HiconaCooler instance."""

        bins: "PlChunks" = handle.get_bins(
            region,
            df_dtype="polars",
            as_chunks=True,
            chunk_size=store_size,
        )
        pixels: "PlChunks" = handle.get_pixels(
            region,
            df_dtype="polars",
            as_chunks=True,
            chunk_size=store_size,
        )
        info = {"region": region}  # TODO: decide what goes in info

        return cls(bins=bins, pixels=pixels, info=info, store_size=store_size)

    def subset(self, region: str, *, store_size: int = 10_000_000) -> "HiconaTable":
        """Return a new HiconaTable instance with data from a specific region."""
        return HiconaTable(
            bins=self.get_bins(region=region, chunk_size=store_size),
            pixels=self.get_pixels(region=region, chunk_size=store_size),
            info=self._info,
            store_size=store_size,
        )

    # def write_cooler(self, path: str):
    #     """Write pixels and bins to a new cooler file."""

    # def plot_matrix(self):
    #     """Plot the genomic region data as a contact matrix."""

    # def jaccard_index(self, other: "HiconaTable"):
    #     """Calculate the Jaccard index between two tables."""
