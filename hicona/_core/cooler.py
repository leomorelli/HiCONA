"""
Custom cooler file handle to extend cooler.Cooler class functionalities.

HiconaCooler inherits from cooler.Cooler and add some functionalities to it,
mainly the ability to manipulate bin annotations, which are severly limited in
use in the original class.

"""

from math import ceil
from os import path
from typing import TYPE_CHECKING, Literal, Union, cast

import cooler
import h5py  # type: ignore
import pandas as pd
import polars as pl

from .._utils.chunked_ops import (
    add_ind_col,
    cast_dtypes,
    convert,
    row_filter,
)
from ._bin_annotation import get_annotated_bins
from .table import BinTable, PixelTable

if TYPE_CHECKING:
    from .._utils.df_dtypes import DataFrame, PlChunks


__all__ = ["HiconaCooler"]
BASE_BIN_COLS: tuple[str, str, str] = ("chrom", "start", "end")
AnnoMetric = Literal["bp_overlap", "frac_overlap", "chrom_enrich"]


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
            raise KeyError(f"An column called {data.name} already exists.")

        # h5py does not recognize polars dtypes, so convert to pandas before.
        # String columns are still not recognized since they become of type "O".
        # For string columns you need to create a fixed lenght string dtype.
        # Check is done on polars to be sure it is a string, since "O" is generic.
        dtype = (
            data.to_pandas().dtype
            if data.dtype != pl.String
            else h5py.string_dtype(length=data.fill_null("None").str.len_chars().max())
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
    """Cooler file handler with extended functionalities.

    Parameters
    ----------
    store : str, :py:class:`h5py.File` or :py:class:`h5py.Group`
        Path to a cooler file, URI string, or open handle to the root HDF5
        group of a cooler data collection. If providing a group via string,
        use the :file:`<file_path>::<group_path>` format.
    kwargs : optional
        Options to be passed to :py:class:`h5py.File()` upon every access.
        By default, the file is opened with the default driver and mode='r'.


    For more information on the class constructor, see :py:class:`cooler.Cooler`.

    """

    def __init__(self, store: Union[str, "h5py.File", "h5py.Group"], **kwargs):
        # Mask deprecated root parameter from super-class
        super().__init__(store, **kwargs)

    def get_bin_table(self, *, store_size: int = 10_000_000) -> "BinTable":
        """Return a bin table containing all bins from the cooler.

        Return an instance of the :py:class:`BinTable` class containing the
        entire bin table from the cooler (annotations included).
        See :py:class:`BinTable` class documentation for more information.

        Parameters
        ----------
        store_size : int, optional
            Max number of bins per parquet storage chunk. Default is ``10_000_000``.

        Returns
        -------
        :py:class:`BinTable`
            Bin table handler.

        """

        chunks: "PlChunks" = _chunked_selector(self.bins(), store_size)
        chunks = add_ind_col(chunks, "bin_id")
        chunks = cast_dtypes(chunks, {"chrom": str})

        return BinTable(chunks, store_size=store_size)

    def get_pixel_table(
        self,
        region: str | None = None,
        *,
        store_size: int = 10_000_000,
    ) -> "PixelTable":
        """Return a pixel table containing all or part of the pixels in the cooler.

        Return an instance of the :py:class:`PixelTable` class, containing all or part
        of the pixels from the cooler. For the fetched pixels, all annotations are kept,
        if any is present.
        See :py:class:`PixelTable` class documentation for more information.

        Parameters
        ----------
        region : str, optional
            Genomic region of interest in the format ``chr:start-end`` or ``chr``. If
            not provided, fetch all pixels. Default is ``None``.
        store_size : int, optional
            Max number of pixels per parquet storage chunk. Default is ``10_000_000``.

        Returns
        -------
        :py:class:`PixelTable`
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
            bins=self.get_bin_table(store_size=store_size),
            store_size=store_size,
        )

    def bin_annot_add(
        self,
        annot_df: "DataFrame",
        *,
        metric: AnnoMetric = "frac_overlap",
        consolidate: bool = True,
        save_all_mods: bool = False,
        ignore_null_mode: bool | Literal["auto"] = "auto",
    ) -> None:
        """Add a new bin annotation column to the bin table.

        Given a dataframe containing some annotation in bed-like format, intersect
        it with the bin table and store it in the cooler as a new bin table column.

        Parameters
        ----------
        annot_df: pandas.DataFrame or polars.DataFrame
            A bed-like dataframe to merge to the bin table. The dataframe must contain
            the columns ``chrom``, ``start``, ``end`` and 1 annotation column.
        metric: one of {``bp_overlap``, ``chrom_enrich``, ``frac_overlap``}, optional
            In case of multiple intersections with a bin, metric used to decide which
            intersection to keep. Available strategies are:

            - ``bp_overlap``: keep the intersection with highest overlap in base pairs.
            - ``frac_overlap``: same as ``bp_overlap`` but as fraction of bin size.
            - ``chrom_enrich``: Requires a categorical annotation covering the entire
              genome (initially designed for chromHMM-style annotations). For each
              bin, assign the modality which is most enriched with respect to its
              own chromosome. Enrichment for a modality is computed as the log2 fold
              change between the fraction of bp in the bin assigned to the modality
              and the fraction of bp in the chromosome (containing the bin) assigned to
              the modality. Only available if ``consolidate = True``.

            Default is ``frac_overlap``.
        consolidate: bool, optional
            When the annotation column is categorical with few repetitive modalities,
            if set to ``True``, all intersections belonging to the same modality are
            considered jointly, summing all overlaps of the modality across the bin.
            Default is ``True``.
        save_all_mods: bool, optional
            When the annotation column is categorical with few repetitive modalities,
            if set to ``True``, instead of choosing the best modality for each bin
            according to the selected metric, create a column for each modality and
            save the metric for each modality for each bin. Only available if
            ``consolidate = True``. Default is ``False``.
        ignore_null_mode: bool or ``auto``, optional
            Whether to consider no annotation (null) as an annotation modality. If
            ``True``, if the null modality is the one with the highest value according
            to the chosen metric, it will be chosen for the annotation. In the same
            scenario, if ``ignore_null_mode = False``, the modality with the second
            highest value is chosen (if available, else null). ``auto`` defaults to
            ``False`` if the metric is an enrichment, to ``True`` otherwise. This
            parameter is ignored if ``save_all_mods = True``. Default is ``auto``.

        Warning
        -------
        Adding annotations can be quite expensive in terms of file size. Use this
        functionality sparingly and only for annotations you are likely to use often.
        For one-time use annotations consider annotating the :py:class:`BinTable` or
        :py:class:`PixelTable` themselves.

        Note
        ----
        Currently it is assumed that the entire bin table fits into memory. If extremely
        small resolutions (and therefore large bin tables) become mainstay, the function
        will be changed to work in chunks.

        """

        # Only fetch bare bins to avoid erroneous splits on already saved columns
        bins_df: pd.DataFrame = cast(pd.DataFrame, self.bins()[:])
        bins_df = bins_df[[*BASE_BIN_COLS]]

        annot_df = get_annotated_bins(
            bins_df,
            annot_df,
            metric,
            consolidate,
            save_all_mods,
            ignore_null_mode,
        )

        for col in annot_df:
            if col.name in BASE_BIN_COLS:
                continue

            _hdf5_writer(self.store, path.join(self.root, "bins"), col)

    def bin_annot_del(self, annot_name: str) -> None:
        """Delete a bin annotation column from the bin table.

        Permanently remove a bin annotation column present from the bin table,
        as long as it is not one of the default ones (e.i. ``chrom``, ``start``,
        ``end``)

        Parameters
        ----------
        annot_name: str
            Name of the annotation to remove from the bin table.

        Warning
        -------
        Due to the ``hdf5`` file format works, deleting a dataset only means removing
        the link to it; the dataset is actually still there, just not reachable. To
        actually reduce the dimension of the file, it is currently necessary to use
        an external tool, such as ``h5repack``.

        """

        if annot_name in BASE_BIN_COLS:
            raise ValueError("Cannot remove a base column (`chrom`, `start`, `end`).")

        _hdf5_deleter(self.store, path.join(self.root, "bins"), annot_name)
