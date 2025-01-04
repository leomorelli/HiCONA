"""
Custom cooler file handle to extend cooler.Cooler class functionalities.

HiconaCooler inherits from cooler.Cooler and add some functionalities to it,
mainly the ability to manipulate bin annotations, which are severly limited in
use in the original class.

"""

from os import path
from math import ceil
from typing import cast, Literal, TYPE_CHECKING, Union

import cooler
import h5py  # type: ignore
import pandas as pd
import polars as pl

from .._utils.chunked_ops import (
    add_ind_col,
    convert,
    row_filter,
    cast_dtypes,
)
from .table import BinTable, PixelTable
from ._bin_annotation import get_annotated_bins

if TYPE_CHECKING:
    from .._utils.df_dtypes import PlChunks, DataFrame


__all__ = ["HiconaCooler"]
BASE_BIN_COLS: tuple[str, str, str] = ("chrom", "start", "end")


def _chunked_selector(selector, chunk_size) -> "PlChunks":
    """Take a RangeSelector1D from Cooler and return it in chunks."""

    num_chunks: int = ceil(len(selector) / chunk_size)
    for i in range(num_chunks):
        start: int = i * chunk_size
        stop: int = min((i + 1) * chunk_size, len(selector))
        yield pl.from_pandas(selector[start:stop])


def _hdf5_writer(store: str, path: str, data: pl.Series) -> None:
    """Write a polars series as a column-like dataset in a .hdf5 file."""

    with h5py.File(store, mode="r+") as h5_handle:
        group: h5py.Group = h5_handle.require_group(path)

        if data.name in group.keys():
            raise KeyError(f"An annotation called {data.name} already exists.")

        # h5py does not recognize polars dtypes, so convert to pandas before.
        # String columns are still not recognized since they become of type "O".
        # For string columns you need to create a fixed lenght string dtype.
        # Check is done on polars to be sure it is a string, since "O" is generic.
        dtype = (
            data.to_pandas().dtype
            if data.dtype != pl.String
            else h5py.string_dtype(length=data.str.len_chars().max())
        )

        group.create_dataset(
            data.name,
            data=data.to_pandas(),
            compression="gzip",
            dtype=dtype,
        )


def _hdf5_deleter(store: str, path: str, df_name: str) -> None:
    """Delete a dataset from an hdf5 file."""

    with h5py.File(store, mode="r+") as h5_handle:
        group: h5py.Group = h5_handle.require_group(path)

        if df_name not in group.keys():
            raise KeyError(f"Annotation `{df_name}` does not exist.")

        del group[df_name]


class HiconaCooler(cooler.Cooler):  # type: ignore  # Cooler is not exported explicitly
    """Cooler file handle with extended functionalities.

    For class constructor documentation, see :class:`cooler.Cooler`.
    # TODO: Copy docs from cooler.Cooler or write new ones.
    """

    def __init__(self, store: Union[str, "h5py.File", "h5py.Group"], **kwargs):
        # Mask deprecated root parameter from super-class
        super().__init__(store, **kwargs)

    def get_bins(
        self,
        region: str | None = None,
        *,
        store_size: int = 10_000_000,
    ) -> "BinTable":
        """Returns a bin table handler.

        Returns an instance of :class:`hicona.BinTable`, potentially subsetted to a
        genomic region of interest.

        Parameters
        ----------
        region : str, optional
            Genomic region of interest in the format "chr:start-end" or "chr".
        store_size : int, optional
            Max number of bins per parquet storage chunk. Default is 10_000_000.

        Returns
        -------
        BinTable
            Bin table handler.

        Notes
        -----
        For small bin tables, which is often the case, this method is algorithmically
        more expensive than using the ``bins()`` method from the cooler.Cooler class.
        The advantage of this method is that it allows to read in chunks large bin
        tables, which could be the case for very high resolution Hi-C data. Moreover,
        for small bin tables, the difference in performance should be negligible but
        with the advantage of being able to return the bins as polars.DataFrame objects.

        """

        chunks: "PlChunks" = _chunked_selector(self.bins(), store_size)
        chunks = add_ind_col(chunks, "bin_id")

        if region:  # Slightly more efficient than subsetting the table
            lower, upper = self.extent(region)
            expr = (pl.col("bin_id") >= lower) & (pl.col("bin_id") < upper)
            chunks = row_filter(chunks, expr, store_size)

        chunks = cast_dtypes(chunks, {"chrom": str})
        return BinTable(chunks, store_size=store_size)

    def get_pixels(
        self,
        region: str | None = None,
        *,
        store_size: int = 10_000_000,
    ) -> "PixelTable":
        """Returns a pixel table handler.

        Returns an instance of :class:`hicona.PixelTable`, potentially subsetted to a
        genomic region of interest.

        Parameters
        ----------
        region : str, optional
            Genomic region of interest in the format "chr:start-end" or "chr".
        store_size : int, optional
            Max number of pixels per parquet storage chunk. Default is 10_000_000.

        Returns
        -------
        PixelTable
            Pixel table handler.

        """

        chunks: "PlChunks" = _chunked_selector(self.pixels(), store_size)

        if region:  # Slightly more efficient than subsetting the table
            lower, upper = self.extent(region)
            bin1_bounds = (pl.col("bin1_id") >= lower) & (pl.col("bin1_id") < upper)
            bin2_bounds = (pl.col("bin2_id") >= lower) & (pl.col("bin2_id") < upper)
            expr = bin1_bounds & bin2_bounds

            chunks = row_filter(convert(chunks, "polars"), expr, store_size)

        return PixelTable(
            chunks,
            bins=self.get_bins(region, store_size=store_size),
            store_size=store_size,
        )

    def bin_annot_add(
        self,
        annot_df: "DataFrame",
        *,
        metric: Literal["bp_overlap", "chrom_enrich", "frac_overlap"] = "frac_overlap",
        consolidate: bool = True,
        save_all_mods: bool = False,
    ) -> None:
        """Add a new bin annotation column to the bin table in the file.

        Given a dataframe containing some annotation in bed-like format, intersect
        it with the bin table and store it in the cooler as a new bin table column.

        Parameters
        ----------
        annot_df: pandas.DataFrame or polars.DataFrame
            A bed-like dataframe to merge to the bin table. The dataframe must contain
            the columns "chrom", "start", "end" and 1 annotation column.
        metric: one of ["bp_overlap", "chrom_enrich", "frac_overlap"]
            In case of multiple intersections with a bin, metric used to decide which
            intersection to keep. Available strategies are:

            - `bp_overlap`: keep the intersection with highest overlap in base pairs.
            - `frac_overlap`: same as `bp_overlap` but as fraction of bin size.
            - `chrom_enrich`: Requires a categorical annotation covering the entire
              genome (initially designed for chromHMM style annotations). For each
              bin, assign the modality which is most enriched with respect to its
              own chromosome. Enrichment for a modality is computed as the log2 fold
              change between the fraction of bp in the bin assigned to the annotation
              and the fraction of bp in the chromosome containing the bin assigned to
              the annotation. Only available if `consolidate = True`.

        consolidate: bool
            When the annotation column is categorical with few repetitive modalities,
            if set to `True`, all intersections belonging to the same modality are
            considered jointly, summing all overlaps of the modality across the bin.
        save_all_mods: bool
            When the annotation column is categorical with few repetitive modalities,
            if set to `True`, instead of choosing the best modality for each bin
            according to the selected metric, create a column for each modality and
            save the metric for each modality for each bin. Only available if
            `consolidate = True`.

        Warning
        -------
        Adding annotations can be quite expensive in terms of file size. Use this
        functionality sparingly and only for annotations you are likely to use often.
        For one-time use annotations consider annotating the `BinTable` itself.

        Note
        ----
        Currently it is assumed that the entire bin table fits into memory. If extremely
        small resolutions (and therefore large bin tables) become mainstay, the function
        will be changed to work in chunks.

        """

        # Only fetch bare bins to avoid erroneous splits on alreadyt saved columns
        bins_df: pd.DataFrame = cast(pd.DataFrame, self.bins()[:])
        bins_df = bins_df[[*BASE_BIN_COLS]]

        annot_df = get_annotated_bins(
            bins_df,
            annot_df,
            metric,
            consolidate,
            save_all_mods,
        )

        for col in annot_df:
            if col.name in BASE_BIN_COLS:
                continue

            _hdf5_writer(self.store, path.join(self.root, "bins"), col)

    def bin_annot_del(self, annot_name: str) -> None:
        """Delete a bin annotation column from the bin table in the file.

        Parameters
        ----------
        annot_name: str
            Name of the annotation to remove from the bin table.

        Warning
        -------
        Due to the `hdf5` file format works, deleting a dataset only means removing
        the link to it; the dataset is actually still there, just not reachable. To
        actually reduce the dimension of the file, it is currently necessary to use
        an external tool, such as `h5repack`.
        """

        if annot_name in BASE_BIN_COLS:
            raise ValueError("Cannot remove a base column (`chrom`, `start`, `end`).")

        _hdf5_deleter(self.store, path.join(self.root, "bins"), annot_name)

    # TODO: Implement the following methods
    # def get_graph(self, region: str | None) -> "HiconaGraph":
    #     """Create and return an instance of HiconaGraph."""
    #     return HiconaGraph.from_cooler(self, region=region)
