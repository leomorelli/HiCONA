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
from typing import Any, cast, Iterable, Literal, overload, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
import polars as pl

from .._utils.chunked_ops import rechunk, convert
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

MatrixMode = Literal["upper", "lower", "full"]
BASE_BIN_COLS: tuple[str, str, str] = ("chrom", "start", "end")


class Table(TmpStorage):
    """Parquet table saved in a temporary storage.

    Creates a folder in the system's temporary directory to store the table data.
    The folder is deleted when execution ends or the kernel is killed.
    The table is saved in parquet format and can be accessed as a generator of chunks.

    Parameters
    ----------
    prefix : str
        Prefix for the temporary storage folder name.
    chunk_size : int, optional
        Max number of rows per parquet storage chunk. Default is 10_000_000.

    """

    def __init__(self, prefix: str, chunk_size: int = 10_000_000):
        super().__init__(prefix)
        self._chunk_size = chunk_size

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

        for i, chunk in enumerate(rechunk(chunks, self._chunk_size)):
            chunk.write_parquet(
                self.tmp_store / f"chunk_{str(i).zfill(4)}.parquet",
                statistics=False,
            )


class BinTable(Table):
    """Handler for bin data stored in a temporary folder.

    The provided data is saved into a temporary parquet folder and can be accessed
    as a dataframe or a generator of chunks. This is significantly faster when only
    a subset of the data is needed and needs to be iterated multiple times.

    Parameters
    ----------
    bins : polars.DataFrame, pandas.DataFrame or a interable of either.
        The bin data to save in the temporary storage.
    store_size : int, optional
        Max number of rows per parquet storage chunk. Default is 10_000_000.

    """

    def __init__(self, bins: "DfStream", store_size: int = 10_000_000):
        super().__init__("hicona_bins", store_size)
        self._save_chunks(convert(bins, "polars"))

        resolution: None | int = None
        for chunk in self._get_chunks():
            row_dict: dict[str, str | int] = chunk.row(0, named=True)
            resolution = int(row_dict["end"]) - int(row_dict["start"])
            break
        assert resolution is not None
        self._resolution: int = resolution

    @property
    def resolution(self) -> int:
        """Return the resolution of the bins."""
        return self._resolution

    def extent(self, region: str) -> tuple[int, int]:
        """Return the lower and upper bin ids for a genomic region of interest.

        Parameters
        ----------
        region : str
            Genomic region of interest in the format "chr:start-end" or "chr".

        Returns
        -------
        tuple[int, int]
            Lower and upper bin ids for the region.

        """

        def bins_filter(region: str, res: int) -> pl.Expr:
            """Filter to apply on the bins to find boundary ids."""
            parts = region.split(":")
            bins_expr = pl.col("chrom") == parts[0]

            if len(parts) == 2:
                start, end = map(int, parts[1].split("-"))
                start = (start // res) * res
                end = (end // res + 1 if end % res else end // res) * res
                bins_expr &= (pl.col("start") >= start) & (pl.col("end") <= end)

            return bins_expr

        lower_id: int | None = None
        upper_id: int | None = None
        bin_expr: pl.Expr = bins_filter(region, self._resolution)

        # Filter the chunk and act according to how many rows are left
        # This flowchart assumes that the bins are sorted and contiguous.
        # TODO: maybe enforce that the bins are sorted and contiguous
        # 0 rows:
        #    - If the upper bound is set, you overshot the region and can stop.
        #    - If the lower bound is not set, you are before the region.
        # 0 < rows < chunk_size:
        #    - You found the top of the region, set the upper bound.
        # chunk_size rows:
        #    - It might be the top of the region, but cannot be sure. Continue.
        for chunk in self._get_chunks():
            chunk = chunk.filter(bin_expr)

            if chunk.height == 0:
                if upper_id is not None:
                    break
                continue

            if lower_id is None:
                lower_id = int(chunk.get_column("bin_id").min())  # type: ignore
            upper_id = int(chunk.get_column("bin_id").max())  # type: ignore

        assert lower_id is not None and upper_id is not None
        return lower_id, upper_id

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

        bin_filter: pl.Expr | None = None
        if region:
            lower, upper = self.extent(region)
            bin_filter = (pl.col("bin_id") >= lower) & (pl.col("bin_id") < upper)

        df: pl.DataFrame = pl.concat(self._get_chunks(bin_filter))
        return df if dtype == "polars" else df.to_pandas()

    def subset(self, region: str) -> "BinTable":
        """Return a new HiconaTable instance with data from a genomic region.

        Parameters
        ----------
        region : str
            Genomic region of interest in the format "chr:start-end" or "chr".

        Returns
        -------
        BinTable
            A new instance with data from the specified region.

        """
        return BinTable([self.get_dataframe(region)], store_size=self._chunk_size)

    def add_annotation(
        self,
        annot_df: "DataFrame",
        *,
        metric: Literal["bp_overlap", "chrom_enrich", "frac_overlap"] = "frac_overlap",
        consolidate: bool = True,
        save_all_mods: bool = False,
    ) -> "BinTable":
        """Create a new bin table with some annotation column from a bed-like dataframe.

        Given a dataframe containing some annotation in bed-like format, intersect
        it with the BinTable and return a new instance with the annotation added.

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

        new_df: pl.DataFrame = self.get_dataframe().hstack(
            get_annotated_bins(
                self.get_dataframe().select(BASE_BIN_COLS),
                annot_df,
                metric,
                consolidate,
                save_all_mods,
            ).select(pl.exclude(BASE_BIN_COLS))
        )

        return BinTable((new_df,), store_size=self._chunk_size)


class PixelTable(Table):
    """Handler for pixel data stored in a temporary folder.

    The provided data is saved into a temporary parquet folder and can be accessed
    as a dataframe or a generator of chunks. This is significantly faster when only
    a subset of the data is needed and needs to be iterated multiple times.

    Parameters
    ----------
    pixels : polars.DataFrame, pandas.DataFrame or a interable of either.
        The pixel data to save in the temporary storage.
    bins : BinTable
        The bin data associated with the pixels.
    store_size : int, optional
        Max number of rows per parquet storage chunk. Default is 10_000_000.

    """

    def __init__(
        self,
        pixels: "DfStream",
        *,
        bins: Union["DfStream", "BinTable"],
        store_size: int = 10_000_000,
    ):
        super().__init__("hicona_pixels", store_size)
        self._save_chunks(convert(pixels, "polars"))
        self._bins = bins if isinstance(bins, BinTable) else BinTable(bins, store_size)

    @property
    def bins(self) -> BinTable:
        """Return the BinTable instance."""
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

        Convert the pixel data, usually stored as a list of edges, into a contact matrix.

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
        selection_kwargs.update({"region": region, "dtype": "polars"})
        df: pl.DataFrame = self.get_dataframe(selection_kwargs=selection_kwargs)

        # Either use left and right bin most ids in the df or the region bounds (for comparison)
        bounds: tuple[int, int]
        if region:
            bounds = self._bins.extent(region)
        else:
            bounds = (
                cast(int, df.get_column("bin1_id").min()),
                cast(int, df.get_column("bin2_id").max()),
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

        # TODO: ideally, speed this up somehow
        # Create an empty matrix with the right dimensions and fill it
        # NOTE: Int conversion is needed since numpy uses a single type for the whole array,
        # and having float value col casts the bin ids to float.
        side: int = bounds[1] - bounds[0] + 1
        # matrix: np.ndarray = np.zeros([side, side], dtype=float)  # TODO: maybe infer
        matrix: np.ndarray = np.empty([side, side], dtype=float)
        matrix.fill(np.nan)
        for row in edge_list:
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
            np.fill_diagonal(matrix, 0)

        return matrix

    def subset(self, region: str) -> "PixelTable":
        """Return a new HiconaTable instance with data from a genomic region.

        Parameters
        ----------
        region : str
            Genomic region of interest in the format "chr:start-end" or "chr".

        Returns
        -------
        PixelTable
            A new instance with data from the specified region.

        """
        return PixelTable(self.get_chunks(region), bins=self._bins.subset(region))

    def apply(self, strategies: Strategy | Iterable[Strategy]) -> "PixelTable":
        """Return a new PixelTable modified according to the provided strategies.

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
            store_size=self._chunk_size,
        )

    def add_bin_annotation(
        self,
        annot_df: "DataFrame",
        *,
        metric: Literal["bp_overlap", "chrom_enrich", "frac_overlap"] = "frac_overlap",
        consolidate: bool = True,
        save_all_mods: bool = False,
    ) -> None:
        """Add some annotation columns from a bed-like dataframe to the bin table.

        Given a dataframe containing some annotation in bed-like format, intersect
        it and add the information to the bin table.

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
        )

    # TODO: from_graph
    # TODO: from_cooler
