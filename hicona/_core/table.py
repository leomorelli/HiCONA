"""
Class to handle data subsets from a cooler file which are temporarily saved to disk.

HiconaTable creates two temporary storages, one for the bins and one for the pixels.
These storages are used to store the data from a specific region of the cooler file
without having to iterate through (and decompress) the whole file every time.
Temporary storages are torn down when the instance is deleted.

"""

from __future__ import annotations

import os
from typing import cast, Literal, overload, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
import polars as pl

from .._utils.chunked_ops import rechunk, convert
from .._utils.tmp_storage import TmpStorage

if TYPE_CHECKING:
    from .._utils.df_dtypes import (
        DataFrame,
        DfChunks,
        DfStream,
        PdChunks,
        PlChunks,
        PlStream,
        DfDtype,
    )

__all__ = ["BinTable", "PixelTable"]

MatrixMode = Literal["upper", "lower", "full"]


def _annotate(pixels: pl.DataFrame, bins: pl.DataFrame) -> pl.DataFrame:
    """Annotate the pixel data with bin information."""

    return (
        pixels.join(bins, how="left", left_on="bin1_id", right_on="bin_id")
        .join(bins, how="left", left_on="bin2_id", right_on="bin_id", suffix="2")
        .rename({c: c + "1" for c in bins.columns if c != "bin_id"})
    )


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
    def get_dataframe(self, region: str | None = ...) -> pl.DataFrame:
        ...

    @overload
    def get_dataframe(
        self, region: str | None = ..., *, dtype: Literal["polars"]
    ) -> pl.DataFrame:
        ...

    @overload
    def get_dataframe(
        self, region: str | None = ..., *, dtype: Literal["pandas"]
    ) -> pd.DataFrame:
        ...

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
        self, region: str | None = ..., *, annotate: bool = ...
    ) -> pl.DataFrame:
        ...

    @overload
    def get_dataframe(
        self,
        region: str | None = ...,
        *,
        annotate: bool = ...,
        dtype: Literal["polars"],
    ) -> pl.DataFrame:
        ...

    @overload
    def get_dataframe(
        self,
        region: str | None = ...,
        *,
        annotate: bool = ...,
        dtype: Literal["pandas"],
    ) -> pd.DataFrame:
        ...

    def get_dataframe(
        self,
        region: str | None = None,
        *,
        annotate: bool = False,
        dtype: DfDtype = "polars",
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

        Returns
        -------
        polars.DataFrame or pandas.DataFrame
            The pixels as a dataframe.

        """

        chunks: "PlChunks" = self.get_chunks(region, annotate=annotate, dtype="polars")
        df: pl.DataFrame = pl.concat(chunks)  # TODO: fix error on concat empty list
        return df if dtype == "polars" else df.to_pandas()

    @overload
    def get_chunks(
        self, region: str | None = ..., *, annotate: bool = ..., chunk_size: int = ...
    ) -> "PlChunks":
        ...

    @overload
    def get_chunks(
        self,
        region: str | None = ...,
        *,
        annotate: bool = ...,
        chunk_size: int = ...,
        dtype: Literal["polars"],
    ) -> "PlChunks":
        ...

    @overload
    def get_chunks(
        self,
        region: str | None = ...,
        *,
        annotate: bool = ...,
        chunk_size: int = ...,
        dtype: Literal["pandas"],
    ) -> "PdChunks":
        ...

    def get_chunks(
        self,
        region: str | None = None,
        *,
        annotate: bool = False,
        chunk_size: int = 10_000_000,
        dtype: DfDtype = "polars",
    ) -> "DfChunks":
        """Return the pixels as a generator of chunks.

        Parameters
        ----------
        region : str, optional
            Genomic region of interest in the format "chr:start-end" or "chr".
        annotate : bool, optional
            Whether to annotate the pixel data with bin information. Default is False.
        chunk_size : int, optional
            Max number of rows per chunk. Default is 10_000_000.
        dtype : {"polars", "pandas"}, optional
            Whether to return the chunks as polars or pandas dataframes.
            Default is "polars".

        Returns
        -------
        Generator of polars.DataFrame or pandas.DataFrame
            The pixels as a generator of chunks.

        """

        pix_filter: pl.Expr | None = None
        bins: pl.DataFrame | None = None

        if region:
            lower, upper = self._bins.extent(region)
            pix_filter = (pl.col("bin1_id") >= lower) & (pl.col("bin1_id") < upper)
            pix_filter &= (pl.col("bin2_id") >= lower) & (pl.col("bin2_id") < upper)

        if annotate:
            bins = self._bins.get_dataframe(region)

        chunks = rechunk(self._get_chunks(pix_filter), chunk_size)
        chunks = chunks if bins is None else (_annotate(c, bins) for c in chunks)

        return (chunk.to_pandas() for chunk in chunks) if dtype == "pandas" else chunks

    def get_matrix(
        self,
        region: str | None = None,
        *,
        value_col: str = "count",
        mode: MatrixMode = "full",
        mask_diagonal: bool = False,
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

        Returns
        -------
        numpy.ndarray
            The contact matrix.

        Warning
        -------
        This operation can create a large matrix in memory, use with caution.

        """

        # NOTE: Not using pl.DataFrame.pivot because does not fill missing bin ids.
        df: pl.DataFrame = self.get_dataframe(region, dtype="polars")

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

    # TODO: from_graph
    # TODO: from_cooler
