"""Disk-backed table classes for bin and pixel data.

BinTable and PixelTable store their data in temporary parquet files on disk,
making it possible to iterate over large Hi-C datasets without loading everything
into memory. Temporary storage is torn down when the instance is garbage collected,
when the program exits, or when the Jupyter kernel is shut down.
"""

from __future__ import annotations

import math
import os
from functools import partial
from typing import TYPE_CHECKING, Any, Iterable, Literal, cast, overload

import graph_tool.all as gt
import numpy as np
import pandas as pd
import polars as pl

from .._utils.chunked_ops import add_ind_col, convert, rechunk, to_iterable
from .._utils.tmp_parquet import TmpParquet
from ._bin_annotation import get_annotated_bins
from ._pix_norm import PixNormFunc, norm_functions
from .graph import HiconaGraph
from .strategies import annotate_pixels, balance_pixels, subset_region

if TYPE_CHECKING:
    from .._utils.df_dtypes import (
        DataFrame,
        DfChunks,
        DfDtype,
        DfStream,
        PdChunks,
        PlChunks,
        Strategy,
    )

__all__ = ["BinTable", "PixelTable"]

AnnoMetric = Literal["bp_overlap", "frac_overlap", "chrom_enrich"]
MatrixMode = Literal["upper", "lower", "full"]
BASE_BIN_COLS: tuple[str, str, str] = ("chrom", "start", "end")
BASE_PIX_COLS: tuple[str, str, str] = ("bin1_id", "bin2_id", "count")


class Table:
    """Parquet table saved in a temporary folder.

    Creates a folder in the temporary directory of the system to store the table data.
    The folder is deleted if the object is garbage collected, if the program execution
    ends, or if the Jupyter kernel is shut down. The table is saved in a chunked
    parquet format.

    Methods to read and write chunks are not directly exposed to avoid accidental
    improper usage.

    Parameters
    ----------
    store_size : int, optional
        Max number of rows per parquet storage chunk. Default is ``10_000_000``.

    """

    def __init__(self, store_size: int = 10_000_000):
        self._store = TmpParquet()
        self._store_size = store_size

    def _get_chunks(self, filt_expr: pl.Expr | None = None) -> "PlChunks":
        """Return the table as a generator of chunks."""

        for chunk in self._store.get():
            yield chunk.filter(filt_expr if isinstance(filt_expr, pl.Expr) else True)

    def _get_dataframe(self) -> pl.LazyFrame:
        """Return the full table as a lazy frame."""
        return pl.scan_parquet(self._store.path)

    def _put_chunks(self, chunks: "PlChunks"):
        """Rechunk and write chunks to the parquet store."""
        self._store.put(rechunk(chunks, self._store_size))

    @property
    def store_size(self) -> int:
        """Return the number of rows of the biggest storage chunk.

        The data is stored in a chunked parquet storage. This attribute returns
        the size (in number of rows) of the biggest chunk. This does not directly
        affect the size of the chunks retrieved from the table.

        Returns
        -------
        int
            Number of rows of the biggest storage chunk.

        """
        return self._store_size

    @property
    def col_names(self) -> tuple[str, ...]:
        """Return the names of the columns of the table.

        Returns
        -------
        tuple of strings
            Names of all the columns in the table.

        """
        return tuple(self._store.peek().keys())


class BinTable(Table):
    """Handler for bin data stored in a temporary folder.

    A bin table is a disk backed version of the bins table from a cooler file.
    Bins are copied to disk to facilitate repetitive iteration and manipulation.
    The storage format is a chunked parquet file in a temporary directory.

    It is assumed that the binning satisfies these properties:

        - Bin size is constant among all bins.
        - Bins belonging to the same chromosome are contiguous and sorted.
        - Binning is complete, i.e. it covers the entire reference genome.

    Parameters
    ----------
    bins : polars.DataFrame, pandas.DataFrame or iterable of either
        The bin data to save in the temporary storage.
    store_size : int, optional
        Max number of rows per parquet storage chunk. Default is ``10_000_000``.

    Warning
    -------
    Currently, no check is performed on the validity of the provided bins.
    This is to allow the usage of any assembly for any organism, as well as custom ones.
    You are fully responsible for checking that your binning satisfies the above
    assumptions. Failing to comply might affect your results significantly.

    """

    def __init__(self, bins: "DataFrame | DfStream", store_size: int = 10_000_000):

        super().__init__(store_size)
        self._put_chunks(add_ind_col(convert(to_iterable(bins)[0], "polars"), "bin_id"))

    def __copy__(self) -> "BinTable":
        return BinTable(bins=self.get_dataframe(), store_size=self.store_size)

    def copy(self) -> "BinTable":
        """Return a deep copy of itself.

        A new temporary copy of all the data present in the table is created and the
        handle to it is returned.

        Returns
        -------
        :py:class:`BinTable`
            A deep copy of this object.

        """
        return self.__copy__()

    @property
    def bin_size(self) -> int:
        """Return the bin size in base pairs (resolution).

        It is assumed that all bins have the same size, with the exception of the
        last bin of each chromosome which may be shorter.

        Returns
        -------
        int
            Bin size in base pairs.

        Examples
        --------
        >>> bt.bin_size
        100000

        """

        first_row: dict[str, Any] = self._store.peek()
        return int(first_row["end"]) - int(first_row["start"])

    @overload
    def get_dataframe(self, region: str | None = ...) -> pl.DataFrame: ...

    @overload
    def get_dataframe(
        self, region: str | None = ..., *, dtype: Literal["polars"]
    ) -> pl.DataFrame: ...

    @overload
    def get_dataframe(
        self, region: str | None = ..., *, dtype: Literal["pandas"]
    ) -> pd.DataFrame: ...

    def get_dataframe(
        self,
        region: str | None = None,
        *,
        dtype: DfDtype = "polars",
    ) -> "DataFrame":
        """Return the bins as a dataframe.

        Any annotation column is returned alongside the default ones.

        Parameters
        ----------
        region : str, optional
            Genomic region of interest in the format ``chr:start-end`` or ``chr``. If
            not provided, all bins are returned. Default is ``None``.
        dtype : one of {``polars``, ``pandas``}, optional
            Whether to return the dataframe as a polars or pandas dataframe.
            Default is ``polars``.

        Returns
        -------
        ``polars.DataFrame`` or ``pandas.DataFrame``
            The bins as a dataframe.

        Warning
        -------
        The possibility to subset the binning is given mostly for visualization
        purposes. In general, try to work with a full genome binning, rather than
        a subset of it (subsetting may lead to issues, especially when comparing
        tables).

        Examples
        --------
        >>> bt.get_dataframe()
        shape: (1_000, 4)
        ┌────────┬───────┬────────┬────────┐
        │ bin_id ┆ chrom ┆ start  ┆ end    │
        │ ---    ┆ ---   ┆ ---    ┆ ---    │
        │ i64    ┆ str   ┆ i32    ┆ i32    │
        ╞════════╪═══════╪════════╪════════╡
        │ 0      ┆ chr1  ┆ 0      ┆ 10000  │
        │ 1      ┆ chr1  ┆ 10000  ┆ 20000  │
        │ …      ┆ …     ┆ …      ┆ …      │
        │ 999    ┆ chrX  ┆ 0      ┆ 10000  │
        └────────┴───────┴────────┴────────┘
        >>> bt.get_dataframe("chr1")
        shape: (250, 4)
        ┌────────┬───────┬────────┬────────┐
        │ bin_id ┆ chrom ┆ start  ┆ end    │
        │ ---    ┆ ---   ┆ ---    ┆ ---    │
        │ i64    ┆ str   ┆ i32    ┆ i32    │
        ╞════════╪═══════╪════════╪════════╡
        │ 0      ┆ chr1  ┆ 0      ┆ 10000  │
        │ 1      ┆ chr1  ┆ 10000  ┆ 20000  │
        │ …      ┆ …     ┆ …      ┆ …      │
        │ 249    ┆ chr1  ┆ 90000  ┆ 100000 │
        └────────┴───────┴────────┴────────┘

        """

        filt_expr: pl.Expr | None = None

        if region:
            parts: list[str] = region.split(":")
            filt_expr = pl.col("chrom") == pl.lit(parts[0])

            if len(parts) == 2:
                chrom_range: list[str] = parts[1].split("-")

                if len(chrom_range) != 2:
                    raise ValueError("Genomic range must have one start and one end.")

                try:
                    start, end = map(int, chrom_range)
                except ValueError:
                    raise ValueError("At least one boundary is not convertible to int.")

                filt_expr &= pl.col("start") >= pl.lit(start - self.bin_size + 1)
                filt_expr &= pl.col("end") <= pl.lit(end + self.bin_size - 1)

        df: pl.DataFrame = pl.concat(self._get_chunks(filt_expr))
        return df if dtype == "polars" else df.to_pandas()

    def extent(self, region: str) -> tuple[int, int]:
        """Return the lower and upper bin ids for a genomic region of interest.

        The interval is half-open: the lower bin id is included, the upper is
        excluded (i.e. ``[lower, upper)``).

        Parameters
        ----------
        region : str
            Genomic region of interest in the format ``chr:start-end`` or ``chr``.

        Returns
        -------
        tuple[int, int]
            Lower and upper bin ids for the region.

        Note
        ----
        Unlike :py:func:`cooler.Cooler.extent`, no out of genomic range check is performed.
        If the provided end point of the region falls outside the highest bin for the
        specified chromosome, the highest bin for that chromosome is returned instead.
        No error or warning is raised. A check might be added in the future.

        Examples
        --------
        >>> bt.extent("chr1")
        (0, 250)
        >>> bt.extent("chr1:50000-150000")
        (5, 16)

        """

        # NOTE: Getting the filtered dataframe and extracting min and max from it is
        # definitely not the most efficient way, but it avoids having to lug around
        # chromsizes which could get real messy. If really needed, check lower and
        # upper on individual chunks to avoid materializing the full table in memory.
        df: pl.DataFrame = self.get_dataframe(region)
        lower = df.get_column("bin_id").min()
        upper = df.get_column("bin_id").max()

        # Asserts are split for type checker
        # TODO: Fix unclear return when given a chromosome which is not in the table
        assert isinstance(lower, int)
        assert isinstance(upper, int)

        return (lower, upper + 1)

    def add_annotation(
        self,
        annot_df: "DataFrame",
        *,
        metric: AnnoMetric = "frac_overlap",
        consolidate: bool = True,
        save_all_mods: bool = False,
        ignore_null_mode: bool | Literal["auto"] = "auto",
    ):
        """Add some annotation column from a bed-like dataframe to the :py:class:`BinTable`.

        Given a dataframe containing some annotation in bed-like format, intersect
        it with the :py:class:`BinTable` and add it as new columns.

        Parameters
        ----------
        annot_df : pandas.DataFrame or polars.DataFrame
            A bed-like dataframe to merge to the bin table. The dataframe must contain
            the columns ``chrom``, ``start``, ``end`` and 1 annotation column.
        metric : one of {``bp_overlap``, ``chrom_enrich``, ``frac_overlap``}, optional
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
        consolidate : bool, optional
            When the annotation column is categorical with few repetitive modalities,
            if set to ``True``, all intersections belonging to the same modality are
            considered jointly, summing all overlaps of the modality across the bin.
            Default is ``True``.
        save_all_mods : bool, optional
            When the annotation column is categorical with few repetitive modalities,
            if set to ``True``, instead of choosing the best modality for each bin
            according to the selected metric, create a column for each modality and
            save the metric for each modality for each bin. Only available if
            ``consolidate = True``. Default is ``False``.
        ignore_null_mode : bool or ``auto``, optional
            Whether to consider no annotation (null) as an annotation modality. If
            ``True``, if the null modality is the one with the highest value according
            to the chosen metric, it will be chosen for the annotation. In the same
            scenario, if ``ignore_null_mode = False``, the modality with the second
            highest value is chosen (if available, else null). ``auto`` defaults to
            ``False`` if the metric is an enrichment, to ``True`` otherwise. This
            parameter is ignored if ``save_all_mods = True``. Default is ``auto``.

        Note
        ----
        Currently it is assumed that the entire bin table fits into memory. If extremely
        small resolutions (and therefore large bin tables) become mainstay, the function
        will be changed to work in chunks.

        Examples
        --------
        >>> import polars as pl
        >>> annot = pl.DataFrame({
        ...     "chrom": ["chr1", "chr1", "chr2"],
        ...     "start": [0, 50000, 0],
        ...     "end":   [50000, 100000, 100000],
        ...     "state": ["A", "B", "A"],
        ... })
        >>> bt.add_annotation(annot)
        >>> bt.get_dataframe()
        shape: (1_000, 5)
        ┌────────┬───────┬────────┬────────┬───────┐
        │ bin_id ┆ chrom ┆ start  ┆ end    ┆ state │
        │ ---    ┆ ---   ┆ ---    ┆ ---    ┆ ---   │
        │ i64    ┆ str   ┆ i32    ┆ i32    ┆ str   │
        ╞════════╪═══════╪════════╪════════╪═══════╡
        │ 0      ┆ chr1  ┆ 0      ┆ 10000  ┆ A     │
        │ 1      ┆ chr1  ┆ 10000  ┆ 20000  ┆ A     │
        │ …      ┆ …     ┆ …      ┆ …      ┆ …     │
        │ 999    ┆ chrX  ┆ 0      ┆ 10000  ┆ None  │
        └────────┴───────┴────────┴────────┴───────┘

        """

        # TODO: Add option to make the annotation partial
        # e.i. you can extend the annotation in a second moment (mostly for clustering)

        # Join is needed in case there are previous annotation columns.
        annot_bins = self.get_dataframe().join(
            get_annotated_bins(
                self.get_dataframe().select(BASE_BIN_COLS),
                annot_df,
                metric,
                consolidate,
                save_all_mods,
                ignore_null_mode,
            ),
            how="left",
            on=BASE_BIN_COLS,
        )

        # Fill `null` with `None` string if the annotation is string-like.
        # This is to have a consistent result between tables stored in hdf5 (Nones)
        # and those in parquet (nulls)
        annot_col = [c for c in annot_df.columns if c not in BASE_BIN_COLS][0]
        if annot_col in annot_bins.columns:
            if annot_bins.get_column(annot_col).dtype == pl.String:
                annot_bins = annot_bins.with_columns(
                    pl.col(annot_col).fill_null("None")
                )

        new_store = TmpParquet()
        new_store.put(c for c in (annot_bins,))
        self._store = new_store

    def save(self, path: str) -> None:
        """Save the table to a persistent storage.

        :py:class:`BinTable` instances are stored in the tmp folder and are
        deleted when execution is halted or they go out of scope.
        This method saves the table to a persistent storage from which it
        can be loaded using the :py:func:`BinTable.load` class method.

        Parameters
        ----------
        path : str
            Path where to save the table. Must be a non-existent folder.

        Examples
        --------
        >>> bt.save("path/to/saved_bins")

        """

        if os.path.exists(path):
            raise OSError(f"{path} directory already exists.")

        os.makedirs(path)
        self._store.save(os.path.join(path, "bins"))

    @classmethod
    def load(cls, path: str) -> "BinTable":
        """Load a previously saved table.

        Copies a previously saved :py:class:`BinTable` into the tmp folder
        to be able to further work on it.

        Parameters
        ----------
        path : str
            Path to the previously saved :py:class:`BinTable` instance.

        Returns
        -------
        :py:class:`BinTable`
            A :py:class:`BinTable` instance backed by a copy of the data in
            the tmp folder.

        Note
        ----
        This method creates a tmp copy and does not modify the persistent one.
        If you wish to save changes to the new tmp copy, explicitly save it
        again using the :py:func:`BinTable.save` method.

        Examples
        --------
        >>> bt = hicona.BinTable.load("path/to/saved_bins")
        >>> bt.get_dataframe()
        shape: (1_000, 4)
        ┌────────┬───────┬────────┬────────┐
        │ bin_id ┆ chrom ┆ start  ┆ end    │
        │ ---    ┆ ---   ┆ ---    ┆ ---    │
        │ i64    ┆ str   ┆ i32    ┆ i32    │
        ╞════════╪═══════╪════════╪════════╡
        │ 0      ┆ chr1  ┆ 0      ┆ 10000  │
        │ 1      ┆ chr1  ┆ 10000  ┆ 20000  │
        │ …      ┆ …     ┆ …      ┆ …      │
        │ 999    ┆ chrX  ┆ 0      ┆ 10000  │
        └────────┴───────┴────────┴────────┘

        """

        if not os.path.isdir(path):
            raise OSError(f"{path} is not a valid directory.")

        unexpected = [f for f in os.listdir(path) if f not in ("bins", "pixels")]
        if any(unexpected) or "bins" not in os.listdir(path):
            raise ValueError(f"{path} does not seem to be a bin or pixel table.")

        storage = TmpParquet.load(os.path.join(path, "bins"))
        return BinTable(storage.get())


class PixelTable(Table):
    """Handler for pixel data stored in a temporary folder.

    A pixel table is a disk backed version of the pixel from the cooler file.
    Pixels are copied to disk to facilitate repetitive iteration and
    manipulation (such as subsetting and filtering).
    The storage format is a chunked parquet file in a temporary directory.

    It is assumed that:
        - All bin ids present in the pixel table appear in the bin table
          (or in general, that the bin table contains a full genome binning).
        - Pixels are sorted by ``bin1_id``, then ``bin2_id``.

    Parameters
    ----------
    pixels : polars.DataFrame, pandas.DataFrame or iterable of either
        The pixel data to save in the temporary storage.
    bins : polars.DataFrame, pandas.DataFrame, iterable of either or BinTable
        The bin data associated with the pixels.
    store_size : int, optional
        Max number of rows per parquet storage chunk. Default is ``10_000_000``.

    Warning
    -------
    Currently, no check is performed on the validity of the provided bins and pixels.
    This is to allow the usage of any assembly for any organism, as well as custom ones.
    You are fully responsible for checking that your binning satisfies the above
    assumptions. Failing to comply might affect your results significantly.

    """

    def __init__(
        self,
        pixels: "DataFrame | DfStream",
        *,
        bins: "DataFrame | DfStream | BinTable",
        store_size: int = 10_000_000,
    ):
        super().__init__(store_size)
        self._put_chunks(convert(to_iterable(pixels)[0], "polars"))
        self._bins = bins if isinstance(bins, BinTable) else BinTable(bins, store_size)

    def __copy__(self) -> "PixelTable":
        return PixelTable(
            self.get_chunks(),
            bins=self.bins.get_dataframe(),
            store_size=self.store_size,
        )

    def copy(self) -> "PixelTable":
        """Return a deep copy of itself.

        A new temporary copy of all the data present in the table is created and the
        handle to it is returned.

        Returns
        -------
        :py:class:`PixelTable`
            A deep copy of this object.

        """
        return self.__copy__()

    @property
    def bins(self) -> BinTable:
        """Return the associated :py:class:`BinTable` instance.

        Returns
        -------
        :py:class:`BinTable`
            The associated bin table.

        Examples
        --------
        >>> pt.bins.get_dataframe()
        shape: (1_000, 4)
        ┌────────┬───────┬────────┬────────┐
        │ bin_id ┆ chrom ┆ start  ┆ end    │
        │ ---    ┆ ---   ┆ ---    ┆ ---    │
        │ i64    ┆ str   ┆ i32    ┆ i32    │
        ╞════════╪═══════╪════════╪════════╡
        │ 0      ┆ chr1  ┆ 0      ┆ 10000  │
        │ …      ┆ …     ┆ …      ┆ …      │
        │ 999    ┆ chrX  ┆ 0      ┆ 10000  │
        └────────┴───────┴────────┴────────┘

        """
        return self._bins

    @overload
    def get_dataframe(
        self,
        region: str | None = ...,
        *,
        annotate: bool = ...,
        selection_kwargs: dict[str, Any] | None = ...,
    ) -> pl.DataFrame: ...

    @overload
    def get_dataframe(
        self,
        region: str | None = ...,
        *,
        annotate: bool = ...,
        dtype: Literal["polars"],
        selection_kwargs: dict[str, Any] | None = ...,
    ) -> pl.DataFrame: ...

    @overload
    def get_dataframe(
        self,
        region: str | None = ...,
        *,
        annotate: bool = ...,
        dtype: Literal["pandas"],
        selection_kwargs: dict[str, Any] | None = ...,
    ) -> pd.DataFrame: ...

    def get_dataframe(
        self,
        region: str | None = None,
        *,
        annotate: bool = False,
        dtype: DfDtype = "polars",
        selection_kwargs: dict[str, Any] | None = None,
    ) -> "DataFrame":
        """Return the pixels as a dataframe.

        The table can be subsetted to a genomic region and optionally annotated
        with bin-level information. Further customization can be passed via
        ``selection_kwargs``.

        Parameters
        ----------
        region : str, optional
            Genomic region of interest in the format ``chr:start-end`` or ``chr``.
            If not provided, all pixels are returned. Default is ``None``.
        annotate : bool, optional
            Whether to annotate the pixels with bin information. Default is ``False``.
        dtype : one of {``polars``, ``pandas``}, optional
            Whether to return the dataframe as a polars or pandas dataframe.
            Default is ``polars``.
        selection_kwargs : dict, optional
            Additional arguments passed to the internally called :py:func:`get_chunks`
            method. If explicitly passed, ``region``, ``annotate`` and ``dtype`` will
            override the values declared in this dictionary.

        Returns
        -------
        :py:class:`polars.DataFrame` or :py:class:`pandas.DataFrame`
            The pixels as a dataframe.

        Warning
        -------
        Data is loaded into memory, which can be quite expensive on a non-subsetted table.

        Examples
        --------
        >>> pt.get_dataframe()
        shape: (5_000, 3)
        ┌─────────┬─────────┬───────┐
        │ bin1_id ┆ bin2_id ┆ count │
        │ ---     ┆ ---     ┆ ---   │
        │ i64     ┆ i64     ┆ i32   │
        ╞═════════╪═════════╪═══════╡
        │ 0       ┆ 0       ┆ 142   │
        │ 0       ┆ 1       ┆ 87    │
        │ …       ┆ …       ┆ …     │
        │ 249     ┆ 249     ┆ 201   │
        └─────────┴─────────┴───────┘
        >>> pt.get_dataframe("chr1", annotate=True)
        shape: (5_000, 11)
        ┌─────────┬─────────┬───────┬────────┬───┬────────┬────────┬────────┬─────────┐
        │ bin1_id ┆ bin2_id ┆ count ┆ chrom1 ┆ … ┆ chrom2 ┆ start2 ┆ end2   ┆ weight2 │
        │ ---     ┆ ---     ┆ ---   ┆ ---    ┆   ┆ ---    ┆ ---    ┆ ---    ┆ ---     │
        │ i64     ┆ i64     ┆ i32   ┆ str    ┆   ┆ str    ┆ i32    ┆ i32    ┆ f64     │
        ╞═════════╪═════════╪═══════╪════════╪═══╪════════╪════════╪════════╪═════════╡
        │ 0       ┆ 0       ┆ 142   ┆ chr1   ┆ … ┆ chr1   ┆ 0      ┆ 10000  ┆ 0.00194 │
        │ 0       ┆ 1       ┆ 87    ┆ chr1   ┆ … ┆ chr1   ┆ 10000  ┆ 20000  ┆ 0.00245 │
        │ …       ┆ …       ┆ …     ┆ …      ┆ … ┆ …      ┆ …      ┆ …      ┆ …       │
        │ 249     ┆ 249     ┆ 201   ┆ chr1   ┆ … ┆ chr1   ┆ 90000  ┆ 100000 ┆ 0.00931 │
        └─────────┴─────────┴───────┴────────┴───┴────────┴────────┴────────┴─────────┘

        """

        selection_kwargs = selection_kwargs or {}
        selection_kwargs.update(
            {
                "region": region,
                "annotate": annotate,
                "dtype": "polars",  # Ensure since merging is performed in polars
            }
        )

        chunks: "PlChunks" = self.get_chunks(**selection_kwargs)
        df: pl.DataFrame = pl.concat(chunks)  # TODO: fix error on concat empty list
        return df if dtype == "polars" else df.to_pandas()

    @overload
    def get_chunks(
        self,
        region: str | None = ...,
        *,
        annotate: bool = ...,
        balance: bool = ...,
        chunk_size: int = ...,
        strategies: Strategy | Iterable[Strategy] | None = ...,
    ) -> "PlChunks": ...

    @overload
    def get_chunks(
        self,
        region: str | None = ...,
        *,
        annotate: bool = ...,
        balance: bool = ...,
        chunk_size: int = ...,
        dtype: Literal["polars"],
        strategies: Strategy | Iterable[Strategy] | None = ...,
    ) -> "PlChunks": ...

    @overload
    def get_chunks(
        self,
        region: str | None = ...,
        *,
        annotate: bool = ...,
        balance: bool = ...,
        chunk_size: int = ...,
        dtype: Literal["pandas"],
        strategies: Strategy | Iterable[Strategy] | None = ...,
    ) -> "PdChunks": ...

    def get_chunks(
        self,
        region: str | None = None,
        *,
        annotate: bool = False,
        balance: bool = False,
        chunk_size: int = 10_000_000,
        dtype: DfDtype = "polars",
        strategies: Strategy | Iterable[Strategy] | None = None,
    ) -> "DfChunks":
        """Return the pixels as a generator of chunks.

        Each chunk is a dataframe of the specified type. Chunks can be modified
        by passing default or custom strategies.

        A strategy is any function that takes a generator of :py:class:`polars.DataFrame`
        instances and returns another generator of the same, applying some
        transformation to each chunk in the stream.

        Parameters
        ----------
        region : str, optional
            Genomic region of interest in the format ``chr:start-end`` or ``chr``.
            If not provided, all pixels are returned. Default is ``None``.
        annotate : bool, optional
            Whether to annotate the pixels with bin information. Default is ``False``.
        balance : bool, optional
            Whether to balance count column by bin weights. Default is ``False``.
        chunk_size : int, optional
            Max number of rows per chunk. Default is ``10_000_000``.
        dtype : one of {``polars``, ``pandas``}, optional
            Whether to return the chunks as a polars or pandas dataframes.
            Default is ``polars``.
        strategies : Strategy or Iterable of Strategy, optional
            List of strategies to apply to the chunks. Default is ``None``.

        Returns
        -------
        Generator of :py:class:`polars.DataFrame` or :py:class:`pandas.DataFrame`
            The pixels as a generator of chunks.

        Note
        ----
        Strategies should take only one argument, the generator of chunks, and
        return another generator of chunks. If your function requires more arguments,
        use :py:func:`functools.partial` to create a partial function with the extra
        arguments.

        Examples
        --------
        >>> for chunk in pt.get_chunks("chr1", chunk_size=1000):
        ...     print(chunk.shape)
        (1000, 3)
        (1000, 3)
        (432, 3)

        """

        strats: list[Strategy] = []
        chunks: "DfChunks" = self._get_chunks()
        bins: pl.DataFrame | None = None

        if region:
            strats.append(partial(subset_region, extent=self._bins.extent(region)))

        if balance:
            bins = bins if bins is not None else self._bins.get_dataframe(region)
            strats.append(partial(balance_pixels, bins_df=bins))

        strategies = [strategies] if callable(strategies) else strategies
        strats.extend(strategies or [])  # Add user-defined strategies

        if annotate:
            bins = bins if bins is not None else self._bins.get_dataframe(region)
            strats.append(partial(annotate_pixels, bins_df=bins))

        strats.append(partial(rechunk, size=chunk_size))

        # Apply queue of strategies to the chunks
        for strategy in strats:
            chunks = strategy(chunks)

        # NOTE: Convert is not a strat due to typing issues
        return convert(chunks, dtype)

    def get_matrix(
        self,
        region: str | None = None,
        *,
        value_col: str = "count",
        mode: MatrixMode = "full",
        mask_diagonal: bool = False,
        selection_kwargs: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Return the pixel data as a contact matrix.

        Convert the pixel data, usually stored as a list of edges, into a full
        contact matrix. The matrix can be optionally subsetted to a genomic
        region of interest as well as filtered according to some strategy.

        Parameters
        ----------
        region : str, optional
            Genomic region of interest in the format ``chr:start-end`` or ``chr``.
            If not provided, all pixels are returned. Default is ``None``.
        value_col : str, optional
            Pixel column to use as the value in the matrix. Default is ``count``.
        mode : one of {``upper``, ``lower``, ``full``}, optional
            Whether to return the upper, lower or full matrix. Default is ``full``.
        mask_diagonal : bool, optional
            Whether to mask (= fill with NaNs) the diagonal of the matrix. Default is ``False``.
        selection_kwargs : dict, optional
            Additional arguments to pass to the internally called :py:func:`get_chunks` method.
            If explicitly passed, ``region`` will override the values declared in this dictionary.

        Returns
        -------
        :py:class:`numpy.ndarray`
            The contact matrix.

        Warning
        -------
        This operation can create a huge matrix in memory, use with caution.

        Examples
        --------
        >>> mat = pt.get_matrix("chr1")
        >>> mat.shape
        (250, 250)

        """

        # NOTE: Not using pl.DataFrame.pivot because does not fill missing bin ids.
        selection_kwargs = selection_kwargs or {}
        selection_kwargs.update({"dtype": "polars"})
        df: pl.DataFrame = self.get_dataframe(
            region=region,
            selection_kwargs=selection_kwargs,
        )

        # Either use left and right bin most ids in the df or the region bounds (for comparison)
        bounds: tuple[int, int]
        if region:
            bounds = self._bins.extent(region)

            # NOTE: currently, since the chrom sizes are not preserved, there is no way to
            # verify that the upper boundary is inside the chromosome.
            # To avoid inconsistent matrix sizes without being able to raise a warning,
            # assume that the region is always valid, and extend the table upwards
            # accordingly. There is no way of doing the same for the full chromosomes.
            # TODO: Maybe we should just decide to bring along chrom sizes.
            if len(region.split(":")) > 1:
                start, end = map(int, region.split(":")[1].split("-"))
                resolution = self._bins.bin_size
                expected = math.ceil(end / resolution) - start // resolution

                bounds = (bounds[0], bounds[0] + expected)

        else:
            bounds = (
                cast(int, df.get_column("bin1_id").min()),
                cast(int, df.get_column("bin2_id").max()) + 1,
            )

        # Shift the bin ids to start from 0 and convert to numpy
        edge_list: np.ndarray = (
            df.with_columns(
                pl.col("bin1_id") - bounds[0],
                pl.col("bin2_id") - bounds[0],
            )
            .select(["bin1_id", "bin2_id", value_col])
            .to_numpy()
        )

        # Create an empty matrix with the right dimensions and fill it
        side: int = bounds[1] - bounds[0]
        matrix: np.ndarray = np.empty([side, side], dtype=float)
        matrix.fill(np.nan)

        # TODO: find a way to speed up
        for row in edge_list:
            x, y, val = row
            matrix[int(x), int(y)] = val

        match mode:
            case "upper":
                pass
            case "lower":
                matrix = matrix.T
            case "full":
                nan_mask: np.ndarray = np.isnan(matrix) & np.isnan(matrix.T)
                matrix = np.nan_to_num(matrix, nan=0) + np.nan_to_num(matrix.T, nan=0)
                matrix[nan_mask] = np.nan
                np.fill_diagonal(matrix, np.diag(matrix) / 2)
            case _:
                raise ValueError(f"Invalid mode: {mode}")

        if mask_diagonal:
            np.fill_diagonal(matrix, np.nan)

        return matrix

    def get_graph(self, region: str | None = None) -> HiconaGraph:
        """Return a :py:class:`HiconaGraph` representation of the pixel table.

        Default class constructor arguments are used. For customization,
        instantiate :py:class:`HiconaGraph` directly.

        Parameters
        ----------
        region : str, optional
            Genomic region of interest in the format ``chr:start-end`` or ``chr``.
            If not provided, all bins and pixels are included. Default is ``None``.

        Returns
        -------
        :py:class:`HiconaGraph`
            A graph representation of the pixel table (or part of it).

        Note
        ----
        If no genomic region is provided, the entire binning will be included
        in the graph. This means that even bins without any associated pixel
        are included. Be mindful of it, as it can drastically increase the
        time required to process the graph.

        Examples
        --------
        >>> g = pt.get_graph("chr1")
        >>> g.graph.num_vertices()
        250
        >>> g.graph.num_edges()
        5000

        """
        return HiconaGraph.from_pixel_table(self, region=region)

    def subset(self, region: str) -> "PixelTable":
        """Return a new pixel table with pixels subsetted to a genomic region.

        The bin table is not subsetted. All bin and pixel annotations are
        preserved.

        Parameters
        ----------
        region : str
            Genomic region of interest in the format ``chr:start-end`` or ``chr``.

        Returns
        -------
        :py:class:`PixelTable`
            A new class instance with pixel data from the specified region.

        Examples
        --------
        >>> pt_chr1 = pt.subset("chr1")
        >>> pt_chr1.get_dataframe()
        shape: (5_000, 3)
        ┌─────────┬─────────┬───────┐
        │ bin1_id ┆ bin2_id ┆ count │
        │ ---     ┆ ---     ┆ ---   │
        │ i64     ┆ i64     ┆ i32   │
        ╞═════════╪═════════╪═══════╡
        │ 0       ┆ 0       ┆ 142   │
        │ 0       ┆ 1       ┆ 87    │
        │ …       ┆ …       ┆ …     │
        │ 249     ┆ 249     ┆ 201   │
        └─────────┴─────────┴───────┘

        """
        return PixelTable(self.get_chunks(region), bins=self._bins)

    def apply(self, strategies: Strategy | Iterable[Strategy]) -> "PixelTable":
        """Return a new pixel table modified according to the provided strategies.

        A strategy is any function that takes a generator of :py:class:`polars.DataFrame`
        instances and returns another generator of the same, therefore applying some
        function to each chunk in the stream.

        The binning is not changed in any way.

        Parameters
        ----------
        strategies : Strategy or Iterable of Strategy
            Strategy or list of strategies to apply to the chunks.

        Returns
        -------
        :py:class:`PixelTable`
            A new instance with modified data.

        Note
        ----
        Using this method followed by :py:func:`get_chunks` or :py:func:`get_dataframe`
        on the new instance without passing any strategy yields the same result as
        passing the strategies to the :py:func:`get_chunks` or :py:func:`get_dataframe`
        method of the original instance.
        The key difference is that :py:func:`apply` creates a new instance in memory;
        this means a higher overhead for the first use, but it becomes faster if that
        specific subset needs to be iterated multiple times.

        Examples
        --------
        >>> from functools import partial
        >>> import polars as pl
        >>>
        >>> def distance_filter(chunks, max_dist):
        ...     for chunk in chunks:
        ...         yield chunk.filter(
        ...             (pl.col("bin2_id") - pl.col("bin1_id")) <= max_dist
        ...         )
        >>>
        >>> pt_filtered = pt.apply(partial(distance_filter, max_dist=10))
        >>> pt_filtered.get_dataframe().shape
        (3_200, 3)

        """
        return PixelTable(
            self.get_chunks(strategies=strategies),
            bins=self._bins,
            store_size=self._store_size,
        )

    def add_bin_annotation(
        self,
        annot_df: "DataFrame",
        *,
        metric: AnnoMetric = "frac_overlap",
        consolidate: bool = True,
        save_all_mods: bool = False,
        ignore_null_mode: bool | Literal["auto"] = "auto",
    ) -> None:
        """Add some annotation columns from a bed-like dataframe to the bin table.

        Given a dataframe containing some annotation in bed-like format, intersect
        it with the associated :py:class:`BinTable` and add it as new columns.

        Parameters
        ----------
        annot_df : pandas.DataFrame or polars.DataFrame
            A bed-like dataframe to merge to the bin table. The dataframe must contain
            the columns ``chrom``, ``start``, ``end`` and 1 annotation column.
        metric : one of {``bp_overlap``, ``chrom_enrich``, ``frac_overlap``}, optional
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
        consolidate : bool, optional
            When the annotation column is categorical with few repetitive modalities,
            if set to ``True``, all intersections belonging to the same modality are
            considered jointly, summing all overlaps of the modality across the bin.
            Default is ``True``.
        save_all_mods : bool, optional
            When the annotation column is categorical with few repetitive modalities,
            if set to ``True``, instead of choosing the best modality for each bin
            according to the selected metric, create a column for each modality and
            save the metric for each modality for each bin. Only available if
            ``consolidate = True``. Default is ``False``.
        ignore_null_mode : bool or ``auto``, optional
            Whether to consider no annotation (null) as an annotation modality. If
            ``True``, if the null modality is the one with the highest value according
            to the chosen metric, it will be chosen for the annotation. In the same
            scenario, if ``ignore_null_mode = False``, the modality with the second
            highest value is chosen (if available, else null). ``auto`` defaults to
            ``False`` if the metric is an enrichment, to ``True`` otherwise. This
            parameter is ignored if ``save_all_mods = True``. Default is ``auto``.

        Examples
        --------
        >>> import polars as pl
        >>> annot = pl.DataFrame({
        ...     "chrom": ["chr1", "chr1", "chr2"],
        ...     "start": [0, 50000, 0],
        ...     "end":   [50000, 100000, 100000],
        ...     "state": ["A", "B", "A"],
        ... })
        >>> pt.add_bin_annotation(annot)
        >>> pt.get_dataframe("chr1", annotate=True).select(["state1", "state2", "count"])
        shape: (5_000, 3)
        ┌────────┬────────┬───────┐
        │ state1 ┆ state2 ┆ count │
        │ ---    ┆ ---    ┆ ---   │
        │ str    ┆ str    ┆ i32   │
        ╞════════╪════════╪═══════╡
        │ A      ┆ A      ┆ 142   │
        │ A      ┆ B      ┆ 87    │
        │ …      ┆ …      ┆ …     │
        │ B      ┆ B      ┆ 201   │
        └────────┴────────┴───────┘

        """
        self._bins.add_annotation(
            annot_df,
            metric=metric,
            consolidate=consolidate,
            save_all_mods=save_all_mods,
            ignore_null_mode=ignore_null_mode,
        )

    def add_pix_annotation(self, annot_df: DataFrame) -> None:
        """Add some annotation columns from a pixel-like dataframe to the pixel table.

        Add pixel annotations to the table by merging against both ``bin1_id`` and
        ``bin2_id`` at the same time. The annotation is stored and is automatically
        retrieved when fetching data using :py:func:`get_chunks`, :py:func:`get_dataframe`
        and :py:func:`get_graph`.

        Parameters
        ----------
        annot_df : polars.DataFrame or pandas.DataFrame
            The pixel-like dataframe from which to fetch the annotations.

        Warning
        -------
        The current implementation is unstable and it will likely be changed in the
        future. The main limitations of the current implementation are the following:

        -  The annotation must come from a single dataFrame, which is not ideal for
           large pixel tables.
        -  Partial annotations (maybe due to clustering) cannot be modified.

        Examples
        --------
        >>> import polars as pl
        >>> pix_annot = pl.DataFrame({
        ...     "bin1_id": [0, 0, 1],
        ...     "bin2_id": [0, 1, 1],
        ...     "is_loop": [True, False, True],
        ... })
        >>> pt.add_pix_annotation(pix_annot)
        >>> pt.get_dataframe()
        shape: (5_000, 4)
        ┌─────────┬─────────┬───────┬─────────┐
        │ bin1_id ┆ bin2_id ┆ count ┆ is_loop │
        │ ---     ┆ ---     ┆ ---   ┆ ---     │
        │ i64     ┆ i64     ┆ i32   ┆ bool    │
        ╞═════════╪═════════╪═══════╪═════════╡
        │ 0       ┆ 0       ┆ 142   ┆ true    │
        │ 0       ┆ 1       ┆ 87    ┆ false   │
        │ …       ┆ …       ┆ …     ┆ …       │
        │ 249     ┆ 249     ┆ 201   ┆ false   │
        └─────────┴─────────┴───────┴─────────┘

        """

        if isinstance(annot_df, pd.DataFrame):
            annot_df = pl.from_pandas(annot_df)

        overlap: list[str] = [
            c
            for c in annot_df.columns
            if c not in BASE_PIX_COLS and c in self.col_names
        ]
        if overlap:  # TODO: implement
            raise NotImplementedError(
                "Using tables with column names already present in the pixel table "
                "is currently not supported."
            )

        anno_chunks = (
            c.join(annot_df, on=("bin1_id", "bin2_id")) for c in self._get_chunks()
        )

        new_store = TmpParquet()
        new_store.put(anno_chunks)
        self._store = new_store

    def normalize_counts(
        self,
        norm: Literal["log", "arctan_mean"] | PixNormFunc = "arctan_mean",
        *,
        column: str = "norm_count",
    ) -> None:
        """Apply a count normalization and save the result to a new column.

        Any callable can be used as long as it takes a ``polars.LazyFrame`` and
        a column name string and returns the lazy frame with that column added.

        Provided normalizations are:

        - ``arctan_mean``: ``arctan(count / mean_count) / (pi/2)``
        - ``log``: ``ln(count + 1)``

        Parameters
        ----------
        norm : one of {``arctan_mean``, ``log``} or callable, optional
            Normalization to apply. Default is ``arctan_mean``.
        column : str, optional
            Name of the column to save the results in. Default is ``norm_count``.

        Examples
        --------
        >>> pt.normalize_counts()
        >>> pt.get_dataframe()
        shape: (5_000, 4)
        ┌─────────┬─────────┬───────┬────────────┐
        │ bin1_id ┆ bin2_id ┆ count ┆ norm_count │
        │ ---     ┆ ---     ┆ ---   ┆ ---        │
        │ i64     ┆ i64     ┆ i32   ┆ f64        │
        ╞═════════╪═════════╪═══════╪════════════╡
        │ 0       ┆ 0       ┆ 142   ┆ 0.823      │
        │ 0       ┆ 1       ┆ 87    ┆ 0.712      │
        │ …       ┆ …       ┆ …     ┆ …          │
        │ 249     ┆ 249     ┆ 201   ┆ 0.871      │
        └─────────┴─────────┴───────┴────────────┘

        """

        if column in self.col_names:
            raise ValueError(f"Column `{column}` already exists, use another name.")

        norm_func = norm_functions.get(norm) if isinstance(norm, str) else norm
        if not norm_func:
            raise ValueError(f"`{norm}` is not a valid normalization function.")

        new_store = TmpParquet()
        df = norm_func(self._get_dataframe(), column).collect()
        new_store.put((d for d in df.iter_slices(self._store_size)))
        self._store = new_store

    def add_clustering(
        self,
        region: str | None = None,  # TODO: make mandatory when added inter region
        on: str = "count",
        *,
        marginals: Literal["no", "bins", "pixels"] = "no",
        seed: int = 42,
        logging_level: str = "INFO",  # TODO: create logging level type
    ) -> gt.NestedBlockState:
        """Run hierarchical clustering on the network and store results in the table.

        Clustering results are written back as new columns in both the bin and
        pixel tables. See :py:meth:`HiconaGraph.compute_clustering` for a full
        description of the algorithm and the ``marginals`` options.

        Parameters
        ----------
        region : str, optional
            Genomic region for which to compute the clustering. If not provided,
            the clustering is computed on the full genome. Default is ``None``.
        on : str, optional
            Name of the pixel column to use as edge scores. Default is ``count``.
        marginals : one of {``no``, ``bins``, ``pixels``}, optional
            Which posterior probabilities to compute. Default is ``no``.
        seed : int, optional
            RNG seed for reproducibility. Default is ``42``.
        logging_level : str, optional
            Console log verbosity level (e.g. ``INFO``, ``DEBUG``).
            Default is ``INFO``.

        Returns
        -------
        graph_tool.NestedBlockState
            The last nested block state computed during clustering.

        Examples
        --------
        >>> pt.normalize_counts()
        >>> state = pt.add_clustering("chr1", on="norm_count")
        >>> pt.get_dataframe("chr1")
        shape: (5_000, 5)
        ┌─────────┬─────────┬───────┬────────────┬───────────┐
        │ bin1_id ┆ bin2_id ┆ count ┆ norm_count ┆ level_(0) │
        │ ---     ┆ ---     ┆ ---   ┆ ---        ┆ ---       │
        │ i64     ┆ i64     ┆ i32   ┆ f64        ┆ i32       │
        ╞═════════╪═════════╪═══════╪════════════╪═══════════╡
        │ 0       ┆ 0       ┆ 142   ┆ 0.82       ┆ null      │
        │ 0       ┆ 1       ┆ 87    ┆ 0.71       ┆ 1         │
        │ …       ┆ …       ┆ …     ┆ …          ┆ …         │
        │ 249     ┆ 249     ┆ 201   ┆ 0.84       ┆ 3         │
        └─────────┴─────────┴───────┴────────────┴───────────┘
        >>> pt.bins.get_dataframe("chr1")
        shape: (250, 6)
        ┌────────┬───────┬────────┬────────┬───────────┬───────────┐
        │ bin_id ┆ chrom ┆ start  ┆ end    ┆ level_(0) ┆ level_(1) │
        │ ---    ┆ ---   ┆ ---    ┆ ---    ┆ ---       ┆ ---       │
        │ i64    ┆ str   ┆ i32    ┆ i32    ┆ i32       ┆ i32       │
        ╞════════╪═══════╪════════╪════════╪═══════════╪═══════════╡
        │ 0      ┆ chr1  ┆ 0      ┆ 10000  ┆ 1         ┆ 0         │
        │ 1      ┆ chr1  ┆ 10000  ┆ 20000  ┆ 1         ┆ 0         │
        │ …      ┆ …     ┆ …      ┆ …      ┆ …         ┆ …         │
        │ 249    ┆ chr1  ┆ 90000  ┆ 100000 ┆ 3         ┆ 1         │
        └────────┴───────┴────────┴────────┴───────────┴───────────┘

        """

        def drop_existing_cols(df, all_names, base_names) -> pl.DataFrame:
            """Rm columns who are already present in another df, excluding base ones."""
            to_drop = [c for c in df.columns if c not in base_names and c in all_names]
            return df.drop(to_drop)

        graph: HiconaGraph = self.get_graph(region)
        state = graph.compute_clustering(
            on=on,
            marginals=marginals,
            seed=seed,
            logging_level=logging_level,
        )

        bins: pl.DataFrame = pl.concat(graph.get_bins())
        bins = drop_existing_cols(bins, self.bins.col_names, BASE_BIN_COLS)
        self._bins = BinTable(
            self._bins.get_dataframe().join(bins, on=BASE_BIN_COLS, how="left"),
            store_size=self._bins.store_size,
        )

        # TODO: Below this comment, operations are done on full frames rather than
        # on iterators. This is because there is no easy way to perform full join
        # on chunked tables. This means high memory cost. Try to fix using some
        # some fancy genomic coordinates based merge, look into bioframe

        pixels: pl.DataFrame = pl.concat(graph.get_pixels(keep_genomic=False))
        pixels = drop_existing_cols(pixels, self.col_names, BASE_PIX_COLS)

        # TODO: might need fill_nan with 2 in pix_group
        new_store = TmpParquet()
        new_store.put(
            rechunk(
                [
                    self.get_dataframe()
                    .join(pixels, on=BASE_PIX_COLS, how="full", coalesce=True)
                    .sort(BASE_PIX_COLS)
                ],
                self.store_size,
            )
        )
        self._store = new_store

        return state

    def save(self, path: str) -> None:
        """Save the table to a persistent storage.

        :py:class:`PixelTable` instances are stored in the tmp folder and are
        deleted when execution is halted or they go out of scope.
        This method saves the table to a persistent storage from which it
        can be loaded using the :py:func:`PixelTable.load` class method.

        Parameters
        ----------
        path : str
            Path where to save the table. Must be a non-existent folder.

        Examples
        --------
        >>> pt.save("path/to/saved_pixels")

        """

        self._bins.save(path)  # Let BinTable.save handle validity check
        self._store.save(os.path.join(path, "pixels"))

    @classmethod
    def load(cls, path: str) -> "PixelTable":
        """Load a previously saved table.

        Creates a copy of a previously saved :py:class:`PixelTable` (and associated
        :py:class:`BinTable`) into the tmp folder to be able to further work on it.

        Parameters
        ----------
        path : str
            Path to the previously saved :py:class:`PixelTable` instance.

        Returns
        -------
        :py:class:`PixelTable`
            A :py:class:`PixelTable` instance backed by a copy of the data in the tmp folder.

        Note
        ----
        This method creates a tmp copy and does not modify the persistent one.
        If you wish to save changes to the new tmp copy, explicitly save it
        again using the :py:func:`PixelTable.save` method.

        Examples
        --------
        >>> pt = hicona.PixelTable.load("path/to/saved_pixels")
        >>> pt.get_dataframe()
        shape: (5_000, 3)
        ┌─────────┬─────────┬───────┐
        │ bin1_id ┆ bin2_id ┆ count │
        │ ---     ┆ ---     ┆ ---   │
        │ i64     ┆ i64     ┆ i32   │
        ╞═════════╪═════════╪═══════╡
        │ 0       ┆ 0       ┆ 142   │
        │ 0       ┆ 1       ┆ 87    │
        │ …       ┆ …       ┆ …     │
        │ 249     ┆ 249     ┆ 201   │
        └─────────┴─────────┴───────┘

        """

        if not os.path.isdir(path):
            raise OSError(f"{path} is not a valid directory.")

        unexpected = [f for f in os.listdir(path) if f not in ("bins", "pixels")]
        if any(unexpected) or "pixels" not in os.listdir(path):
            raise ValueError(f"{path} does not seem to be a pixel table.")

        storage = TmpParquet.load(os.path.join(path, "pixels"))
        return PixelTable(storage.get(), bins=BinTable.load(path))
