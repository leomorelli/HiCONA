"""
Class to handle data subsets from a cooler file which are temporarily saved to disk.

HiconaTable creates two temporary storages, one for the bins and one for the pixels.
These storages are used to store the data from a specific region of the cooler file
without having to iterate through (and decompress) the whole file every time.
Temporary storages are torn down when the instance is deleted.

"""

from __future__ import annotations

import os
from functools import partial
from typing import Any, cast, Iterable, Literal, overload, TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl

from .._utils.chunked_ops import add_ind_col, convert, rechunk, to_iterable
from .._utils.tmp_storage import TmpStorage
from ._bin_annotation import get_annotated_bins
from .strategies import annotate_pixels, balance_pixels, subset_region

if TYPE_CHECKING:
    from .._utils.df_dtypes import (
        DataFrame,
        DfChunks,
        DfStream,
        PdChunks,
        PlChunks,
        PlStream,
        DfDtype,
        Strategy,
    )

__all__ = ["BinTable", "PixelTable"]

AnnoMetric = Literal["bp_overlap", "frac_overlap", "chrom_enrich"]
MatrixMode = Literal["upper", "lower", "full"]
BASE_BIN_COLS: tuple[str, str, str] = ("chrom", "start", "end")
BASE_PIX_COLS: tuple[str, str, str] = ("bin1_id", "bin2_id", "count")


class Table(TmpStorage):
    """Parquet table saved in a temporary folder.

    Creates a folder in the temporary directory of the system to store the table data.
    The folder is deleted if the object is garbage collected, if the program execution
    ends, or if the Jupyter kernel is shut down. The table is saved in a chunked
    parquet format.

    Methods to read and write chunks are not directly exposed to avoid accidental
    improper usage.

    Parameters
    ----------
    prefix : str
        Prefix for the temporary storage folder name.
    chunk_size : int, optional
        Max number of rows per parquet storage chunk. Default is 10_000_000.

    """

    def __init__(self, prefix: str, store_size: int = 10_000_000):
        super().__init__(prefix)
        self._store_size = store_size

    def _get_chunks(self, filt_expr: pl.Expr | None = None) -> "PlChunks":
        """Return the table as a generator of chunks."""

        files = os.listdir(self.tmp_store)
        files.sort()

        for file in files:
            yield pl.read_parquet(os.path.join(self.tmp_store, file)).filter(
                filt_expr if isinstance(filt_expr, pl.Expr) else True
            )

    def _save_chunks(self, chunks: "PlStream"):
        """Save the provided chunks into the tmp folder."""

        # TODO: add check that at least one chunk was written
        for i, chunk in enumerate(rechunk(chunks, self._store_size)):
            chunk.write_parquet(
                self.tmp_store / f"chunk_{str(i).zfill(4)}.parquet",
                statistics=False,
            )

    def _peek(self) -> dict[str, Any]:
        """Return the first row of the dataframe as a dictionary."""

        row_dict: dict[str, Any] | None = None
        for chunk in self._get_chunks():
            row_dict = chunk.row(0, named=True)
            break

        assert row_dict is not None, f"Table at {self.tmp_store} is empty."
        return row_dict

    @property
    def store_size(self) -> int:
        """Return the max number of rows of the individual store chunks.

        Returns
        -------
        int
            Max number of rows per storage chunk.

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
        return tuple(self._peek().keys())


class BinTable(Table):
    """Handler for bin data stored in a temporary folder.

    A bin table is a disk backed version of the bins from the cooler file.
    Bins are copied to disk to facilitate iteration, avoiding compression.
    The storage format is a chunked parquet file in a temporaty directory.

    It is assumed that the binning satisfies these properties:

        - Bin size is constant among all bins.
        - Bins belonging to the same chromosome are contiguous and sorted.
        - Binning is complete, e.i. it covers the entire reference genome.

    The last point is not fully mandatory, though not satisfying it might
    result in unexpected behavior especially, but not exclusively, when plotting.

    Parameters
    ----------
    bins : polars.DataFrame, pandas.DataFrame or an interable of either.
        The bin data to save in the temporary storage.
    store_size : int, optional
        Max number of rows per parquet storage chunk. Default is 10_000_000.

    Warning
    -------
    No check is performed on the validity of the provided bins. This is to allow
    the usage of any assembly for any organism, as well as custom ones. You are
    responsible for checking that your binning satisfies the above assumptions.

    """

    def __init__(self, bins: "DataFrame | DfStream", store_size: int = 10_000_000):
        super().__init__("hicona_bins_", store_size)

        polars_stream: PlStream = convert(to_iterable(bins)[0], "polars")
        self._save_chunks(add_ind_col(polars_stream, "bin_id"))

        # NOTE: bin_size is defined here since it should not change overtime
        first_row: dict[str, Any] = self._peek()
        self._bin_size: int = int(first_row["end"]) - int(first_row["start"])

    @property
    def bin_size(self) -> int:
        """Return the size of the bins in base pairs (resolution).

        Returns
        -------
        int
            Bin size in base pairs.

        """
        return self._bin_size

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
        """Return the bins in a dataframe.

        Return the bins from the table in a dataframe. Any annotation column is
        also returned alongside the default ones.

        Parameters
        ----------
        region : str, optional
            Genomic region of interest in the format "chr:start-end" or "chr".
        dtype : {"polars", "pandas"}, optional
            Whether to return the dataframe as a polars or pandas dataframe.
            Default is "polars".

        Returns
        -------
        polars.DataFrame or pandas.DataFrame
            The bins as a dataframe.

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

                filt_expr &= pl.col("start") >= pl.lit(start - self._bin_size + 1)
                filt_expr &= pl.col("end") <= pl.lit(end + self._bin_size - 1)

        df: pl.DataFrame = pl.concat(self._get_chunks(filt_expr))
        return df if dtype == "polars" else df.to_pandas()

    def extent(self, region: str) -> tuple[int, int]:
        """Return the lower and upper bin ids for a genomic region of interest.

        Unlike `cooler.Cooler.extent`, no out of genomic range check is performed.
        If the end point of the region lays outside a chromosome boundary, the last
        bin of the chromosome is returned as end point.

        Parameters
        ----------
        region : str
            Genomic region of interest in the format "chr:start-end" or "chr".

        Returns
        -------
        tuple[int, int]
            Lower and upper bin ids for the region.

        """

        # NOTE: Getting the filtered dataframe and extracting min and max from it is
        # definitely not the most efficient way, but it avoids having to lug around
        # chromsizes which could get real messy. If really needed, check lower and
        # upper on individual chunks to avoid materializing the full table in memory.
        df: pl.DataFrame = self.get_dataframe(region)
        lower = df.get_column("bin_id").min()
        upper = df.get_column("bin_id").max()

        # Asserts are split for type checker
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
    ) -> "BinTable":
        """Create a new bin table with some annotation column from a bed-like dataframe.

        Given a dataframe containing some annotation in bed-like format, intersect
        it with the BinTable and return a new instance with the annotation added.

        Parameters
        ----------
        annot_df: pandas.DataFrame or polars.DataFrame
            A bed-like dataframe to merge to the bin table. The dataframe must contain
            the columns "chrom", "start", "end" and 1 annotation column.
        metric: one of ["bp_overlap", "chrom_enrich", "frac_overlap"], optional
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

            Default is `frac_overlap`.
        consolidate: bool, optional
            When the annotation column is categorical with few repetitive modalities,
            if set to `True`, all intersections belonging to the same modality are
            considered jointly, summing all overlaps of the modality across the bin.
            Default is True.
        save_all_mods: bool, optional
            When the annotation column is categorical with few repetitive modalities,
            if set to `True`, instead of choosing the best modality for each bin
            according to the selected metric, create a column for each modality and
            save the metric for each modality for each bin. Only available if
            `consolidate = True`. Default is False.
        ignore_null_mode: bool or "auto", optional
                Whether to consider no annotation (null) as an annotation modality. If
                `True`, if the null modality is the one with the highest value according
                to the chosen metric, it will be chosen for the annotation. In the same
                scenarion, if `ignore_null_mode = False`, the modality with the second
                highest value is chosen (if available, else null). `auto` defaults to
                `False` if the metric is an enrichment, to `True` otherwise. This
                parameter is ignored if `save_all_mods = True`. Default is `auto`.


        Returns
        -------
        BinTable
            A new bin table with one (or more) new annotation column(s).

        Note
        ----
        Currently it is assumed that the entire bin table fits into memory. If extremely
        small resolutions (and therefore large bin tables) become mainstay, the function
        will be changed to work in chunks.

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

        return BinTable((annot_bins,), store_size=self._store_size)


class PixelTable(Table):
    """Handler for pixel data stored in a temporary folder.

    A pixel table is a disk backed version of the pixel from the cooler file.
    Pixels are copied to disk to facilitate iteration, avoiding compression,
    as well as subsetting and filtering operations.
    A pixel table can be subsetted to a genomic region of interest.
    The storage format is a chunked parquet file in a temporaty directory.

    It is assumed that:
        - All bin ids present in the pixel table appear in the bin table
        - Pixels belonging to the same chromosome are contiguous and sorted.

    The last point is not fully mandatory, though not satisfying it might
    result in unexpected behavior especially, but not exclusively, when plotting.

    Parameters
    ----------
    pixels : polars.DataFrame, pandas.DataFrame or an interable of either.
        The pixel data to save in the temporary storage.
    bins : BinTable
        The bin data associated with the pixels.
    store_size : int, optional
        Max number of rows per parquet storage chunk. Default is 10_000_000.

    Warning
    -------
    No check is performed on the validity of the provided bins and pixels and
    combination of them. This is to allow the usage of any assembly for any
    organism, as well as custom ones. You are responsible for checking that
    your inputs satisfy the assumptions.

    """

    def __init__(
        self,
        pixels: "DataFrame | DfStream",
        *,
        bins: "DataFrame | DfStream | BinTable",
        store_size: int = 10_000_000,
    ):
        super().__init__("hicona_pixels_", store_size)
        self._save_chunks(convert(to_iterable(pixels)[0], "polars"))
        self._bins = bins if isinstance(bins, BinTable) else BinTable(bins, store_size)

    @property
    def bins(self) -> BinTable:
        """Return the associated bin table.

        Return the instance of the BinTable class associated to this object.

        Returns
        -------
        BinTable
            The associated bin table.

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

        Return the pixels from the table in a dataframe. The table can be subsetted
        to a genomic region of interest. Optionally, the pixels can be annotated using
        all annotation columns present in the pixels.

        Parameters
        ----------
        region : str, optional
            Genomic region of interest in the format "chr:start-end" or "chr".
        annotate : bool, optional
            Whether to annotate the pixel data with bin information. Default is False.
        dtype : {"polars", "pandas"}, optional
            Whether to return the dataframe as a polars or pandas dataframe.
            Default is "polars".
        selection_kwargs : dict, optional
            Additional arguments to pass to the internally called `get_chunks` method.
            If explicitely passed, `region`, `annotate` and `dtype` will override the
            values declared in this dictionary.

        Returns
        -------
        polars.DataFrame or pandas.DataFrame
            The pixels as a dataframe.

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

        Returns a generator of pixel chunks, where each chunk is a dataframe.
        The chunks can be returned as they are or modified using some default
        or custom strategies. A strategy is any function that takes a generator
        of `polars.DataFrame` instances and returns another generator of the same,
        therefore applying some function to each chunk in the stream.

        Note
        ----
        Strategies should take only one argument, the generator of chunks, and
        return another generator of chunks. If your function requires more arguments,
        use `functools.partial` to create a partial function with the extra arguments.

        Parameters
        ----------
        region : str, optional
            Genomic region of interest in the format "chr:start-end" or "chr".
        annotate : bool, optional
            Whether to annotate the pixel data with bin information. Default is False.
        balance : bool, optional
            Whether to balance count column by bin weights. Default is False.
        chunk_size : int, optional
            Max number of rows per chunk. Default is 10_000_000.
        dtype : {"polars", "pandas"}, optional
            Whether to return the chunks as polars or pandas dataframes.
            Default is "polars".
        strategies : Strategy or Iterable of Strategy, optional
            List of strategies to apply to the chunks. Default is None.

        Returns
        -------
        Generator of polars.DataFrame or pandas.DataFrame
            The pixels as a generator of chunks.

        """

        strats: list[Strategy] = []
        chunks: "DfChunks" = self._get_chunks()
        bins: pl.DataFrame | None = None

        if region:
            strats.append(partial(subset_region, extent=self._bins.extent(region)))

        if balance:
            bins = bins or self._bins.get_dataframe(region)
            strats.append(partial(balance_pixels, bins_df=bins))

        strategies = [strategies] if callable(strategies) else strategies
        strats.extend(strategies or [])  # Add user-defined strategies

        if annotate:
            bins = bins or self._bins.get_dataframe(region)
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
            Genomic region of interest in the format "chr:start-end" or "chr".
        value_col : str, optional
            Pixels column to use as the value for the matrix. Default is "count".
        mode : {"upper", "lower", "full"}, optional
            Whether to return the upper, lower or full matrix. Default is "full".
        mask_diagonal : bool, optional
            Whether to mask the diagonal of the matrix. Default is False.
        selection_kwargs : dict, optional
            Additional arguments to pass to the internally called `get_chunks` method.
            If explicitely passed, `region`, will override the values declared in this
            dictionary.

        Returns
        -------
        numpy.ndarray
            The contact matrix.

        Warning
        -------
        This operation can create a large matrix in memory, use with caution.

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
        # NOTE: Int conversion is needed since numpy uses a single type for
        # the whole array, and val is often float, making x, y floats too.
        for row in edge_list:
            x, y, val = row
            matrix[int(row[0]), int(row[1])] = row[2]

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
            np.fill_diagonal(matrix, np.NaN)

        return matrix

    def subset(self, region: str) -> "PixelTable":
        """Return a new pixel table with subsetted data from a genomic region.

        Parameters
        ----------
        region : str
            Genomic region of interest in the format "chr:start-end" or "chr".

        Returns
        -------
        PixelTable
            A new class instance with data from the specified region.

        """
        return PixelTable(self.get_chunks(region), bins=self._bins)

    def apply(self, strategies: Strategy | Iterable[Strategy]) -> "PixelTable":
        """Return a new pixel table modified according to the provided strategies.

        For details on strategies, see the documentation of the `get_chunks` method.

        Parameters
        ----------
        strategies : Strategy or Iterable of Strategy
            Strategy or list of strategies to apply to the chunks.

        Returns
        -------
        PixelTable
            A new instance with modified data.

        Note
        ----
        Using this method followed by `get_chunks` or `get_dataframe` on the new instance
        without passing any strategy yields the same result as passing the strategies to
        the `get_chunks` or `get_dataframe` method of the original instance. The key
        difference is that `apply` creates a new instance in memory; this means a higher
        overhead for the first use, but it becomes faster if that specific subset needs
        to be iterated multiple times.

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
        it and add the information to the bin table.

        Parameters
        ----------
        annot_df: pandas.DataFrame or polars.DataFrame
            A bed-like dataframe to merge to the bin table. The dataframe must contain
            the columns "chrom", "start", "end" and 1 annotation column.
        metric: one of ["bp_overlap", "chrom_enrich", "frac_overlap"], optional
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

            Default is "frac_overlap".
        consolidate: bool, optional
            When the annotation column is categorical with few repetitive modalities,
            if set to `True`, all intersections belonging to the same modality are
            considered jointly, summing all overlaps of the modality across the bin.
            Default is True.
        save_all_mods: bool, optional
            When the annotation column is categorical with few repetitive modalities,
            if set to `True`, instead of choosing the best modality for each bin
            according to the selected metric, create a column for each modality and
            save the metric for each modality for each bin. Only available if
            `consolidate = True`. Default is False.
        ignore_null_mode: bool or "auto", optional
            Whether to consider no annotation (null) as an annotation modality. If
            `True`, if the null modality is the one with the highest value according
            to the chosen metric, it will be chosen for the annotation. In the same
            scenarion, if `ignore_null_mode = False`, the modality with the second
            highest value is chosen (if available, else null). `auto` defaults to
            `False` if the metric is an enrichment, to `True` otherwise. This
            parameter is ignored if `save_all_mods = True`. Default is `auto`.


        Note
        ----
        Currently it is assumed that the entire bin table fits into memory. If extremely
        small resolutions (and therefore large bin tables) become mainstay, the function
        will be changed to work in chunks.

        """

        self._bins = self._bins.add_annotation(
            annot_df,
            metric=metric,
            consolidate=consolidate,
            save_all_mods=save_all_mods,
            ignore_null_mode=ignore_null_mode,
        )

    def add_pix_annotation(self, annot_df: DataFrame) -> None:
        """Add some annotation columns from a bed-like dataframe to the pixel table.

        Add pixel annotations to the table by merging against both `bin1_id` and
        `bin2_id` at the same time. The annotation is stored and is automatically
        retrieved when fetching data using `get_chunks`, `get_dataframe` and
        `get_graph`.

        Parameters
        ----------
        annot_df : polars.DataFrame of pandas.DataFrame
            The bed-like file from which to fetch the annotations.

        Warning
        -------
        The current implementation is unstable and it will likely be changed in the
        future. The main limitations of the current implementation are the following:

        -  The annotation must come from a single DataFrame, which is not ideal for
           large pixel tables.
        -  Partial annotations (maybe due to clustering) cannot be modified
        -  The table is updated directly in its storage, which might lead to
           corruption if the process is halted abruptly.

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
        self._save_chunks(anno_chunks)
        )

