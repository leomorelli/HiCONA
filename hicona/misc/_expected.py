"""Temporary API for observed over expected functionalities."""

import warnings

import cooler
import cooltools as ct
import pandas as pd
import polars as pl

from .._core import HiconaCooler
from .._utils.chunked_ops import convert, to_iterable
from .._utils.df_dtypes import (
    DataFrame,
    DfStream,
    PdChunks,
    PlChunks,
)

__all__ = [
    "expected_normalized_cooler",
    "expected_normalized_pixels",
    "get_expected_counts",
]


def get_expected_counts(handle: HiconaCooler, *, nproc: int = 4) -> pl.DataFrame:
    """Compute cis and trans expected counts for a cooler.

    Compute expected counts for both intra-chromosomal interactions (balanced,
    smoothed, not aggregated) and inter-chromosomal ones (balanced) using
    :py:mod:`cooltools`. Distance is expressed in number of bins; inter-chromosomal
    interactions are assigned a distance of ``-1``. All results are returned in a
    single dataframe.

    Parameters
    ----------
    handle : :py:class:`HiconaCooler`
        Open handle for the cooler for which to compute the expected counts.
    nproc : int, optional
        Number of processes to spawn when using :py:mod:`cooltools`. Default is ``4``.

    Returns
    -------
    polars.DataFrame
        DataFrame with columns ``chrom1``, ``chrom2``, ``dist`` and ``expected``.

    Note
    ----
    Loosely based on code in the `cooltools.sandbox`_.

    .. _cooltools.sandbox: https://github.com/open2c/cooltools/blob/master/cooltools/sandbox/obs_over_exp_cooler.py

    """

    # Fix to https://github.com/open2c/cooltools/issues/509
    NUM_MASKED_DIAG: int = 2
    fix_diag_mask: pl.Expr = (
        pl.when(pl.col("dist") < NUM_MASKED_DIAG)
        .then(None)
        .otherwise(pl.col("expected"))
        .alias("expected")
    )

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        cis: pl.DataFrame = (
            pl.from_pandas(
                ct.expected_cis(
                    handle,
                    intra_only=True,
                    smooth=True,
                    aggregate_smoothed=False,
                    nproc=nproc,
                )
            )
            .rename({"balanced.avg.smoothed": "expected"})
            .with_columns(fix_diag_mask)
            .select("region1", "region2", "dist", "expected")
        )

    trans: pl.DataFrame = (
        pl.from_pandas(ct.expected_trans(handle, nproc=nproc))
        .with_columns(pl.lit(-1).alias("dist").cast(pl.Int64))
        .rename({"balanced.avg": "expected"})
        .select("region1", "region2", "dist", "expected")
    )

    return pl.concat([cis, trans]).rename({"region1": "chrom1", "region2": "chrom2"})


def expected_normalized_pixels(
    pixels: "DataFrame | DfStream",
    expected: DataFrame,
) -> PlChunks:
    """Normalize pixel counts by dividing by the expected count.

    Return pixels where the ``count`` column is replaced by the
    observed-over-expected ratio. Works for both inter- and intra-chromosomal
    contacts.

    Parameters
    ----------
    pixels : polars.DataFrame, pandas.DataFrame or iterable of either
        The pixels to normalize. Must be annotated and balanced.
    expected : polars.DataFrame or pandas.DataFrame
        DataFrame containing the expected counts. Must have columns ``chrom1``,
        ``chrom2``, ``dist`` and ``expected``.

    Returns
    -------
    Generator of polars.DataFrame
        Normalized pixel chunks.

    Warning
    -------
    Due to the way matrix balancing works, some pixels will end up with a count
    of zero or NaN and will be dropped. The number of output pixels will likely
    be smaller than the input.

    Note
    ----
    Loosely based on code in the `cooltools.sandbox`_.

    .. _cooltools.sandbox: https://github.com/open2c/cooltools/blob/master/cooltools/sandbox/obs_over_exp_cooler.py

    """

    pixel_stream: "PlChunks" = convert(to_iterable(pixels)[0], "polars")
    expected = pl.concat(convert((expected,), "polars"))

    pixel_stream = (
        chunk.with_columns(
            pl.when(pl.col("chrom1") == pl.col("chrom2"))
            .then(pl.col("bin2_id") - pl.col("bin1_id"))
            .otherwise(pl.lit(-1))
            .alias("dist")
        )
        .join(expected, on=["chrom1", "chrom2", "dist"])
        .with_columns((pl.col("count") / pl.col("expected")).alias("count"))
        .drop("dist", "expected")
        .filter(pl.col("count") != 0)
        for chunk in pixel_stream
    )

    return pixel_stream


def expected_normalized_cooler(
    handle: HiconaCooler,
    path: str,
    *,
    nproc: int = 4,
) -> pl.DataFrame:
    """Create a new cooler with pixel counts normalized by expected counts.

    Compute expected counts for both intra- and inter-chromosomal interactions
    using :py:mod:`cooltools`, then divide the balanced observed counts by the
    corresponding expected values and write the result to a new cooler file.

    Parameters
    ----------
    handle : :py:class:`HiconaCooler`
        Open handle for the cooler to normalize.
    path : str
        Path where the new normalized cooler file will be written.
    nproc : int, optional
        Number of processes to spawn when using :py:mod:`cooltools`. Default is ``4``.

    Returns
    -------
    polars.DataFrame
        The expected counts used to normalize the file.

    Warning
    -------
    Due to the way matrix balancing works, some pixels will end up with a count
    of zero or NaN and will be dropped. The number of output pixels will likely
    be smaller than the input.

    Note
    ----
    Loosely based on code in the `cooltools.sandbox`_.

    .. _cooltools.sandbox: https://github.com/open2c/cooltools/blob/master/cooltools/sandbox/obs_over_exp_cooler.py

    """

    expected_df: pl.DataFrame = get_expected_counts(handle, nproc=nproc)
    bins: pd.DataFrame = handle.get_bin_table().get_dataframe().to_pandas()
    pixels: PlChunks = handle.get_pixel_table().get_chunks(annotate=True, balance=True)

    # Convert annotated polar chunks into bare pandas chunks
    norm_pixels: PdChunks = (
        c.select("bin1_id", "bin2_id", "count").to_pandas()
        for c in expected_normalized_pixels(pixels, expected_df)
    )

    cooler.create_cooler(path, bins, norm_pixels, dtypes={"count": float})

    return expected_df
