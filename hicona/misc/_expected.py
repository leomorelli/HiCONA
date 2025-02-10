import warnings

import cooler
import cooltools as ct
import pandas as pd
import polars as pl

from .._utils.chunked_ops import convert, to_iterable
from .._utils.df_dtypes import (
    DataFrame,
    DfDtype,
    DfStream,
    PdChunks,
    PlChunks,
)
from .._core import HiconaCooler

__all__ = [
    "expected_normalized_cooler",
    "expected_normalized_pixels",
    "get_expected_counts",
]

REF_LINK = "https://github.com/open2c/cooltools/blob/master/cooltools/sandbox/obs_over_exp_cooler.py"


def get_expected_counts(handle: HiconaCooler, *, nproc: int = 4) -> pl.DataFrame:
    f"""Compute all cis and trans expected counts for a cooler.

    Using cooltools, compute both expected count for intra chromosomal interactions
    (balanced, smoothened, not aggregated) and inter chromosomal ones (balanced).
    Distance is in number of bins. Inter chromosomal interactions are assigned a
    distance of -1. Values are returned in a single dataframe.

    Parameters
    ----------
    handle : HiconaCooler
        Open handle for the cooler for which to compute the expected counts.
    nproc : int
        Number of processes to spawn when using cooltools.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the expected counts.

    Note
    ----
    Loosely based on code in the [cooltools.sandbox]<{REF_LINK}>

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
    *,
    dtype: DfDtype = "polars",
) -> PlChunks:
    f"""Convert observed count values to observed/expected count values.

    Given some annotated pixels, return a pixel table where the count value is
    normalized by dividing by the expected count value. Works for both inter
    and intra chromosomal contacts.

    Parameters
    ----------
    pixels : polars.DataFrame, pandas.DataFrame or an interable of either.
        The pixels to normalize. Must be annotated and balanced.
    expected : polars.DataFrame or pandas.DataFrame
        Dataframe containing the expected counts. Columns must be "chrom1",
        "chrom2", "dist" and "expected".
    dtype : {"polars", "pandas"}, optional
        Whether to return the dataframe as a polars or pandas dataframe.
        Default is "polars".

    Returns
    -------
    polars.DataFrame
        The pixels as a dataframe.

    Warning
    -------
    Due to the way matrix balancing works, some pixels will end up with count
    equal to zero and will thus be dropped. For reason, the number of output
    pixels will likely be smaller than the input one.

    Note
    ----
    Loosely based on code in the [cooltools.sandbox]<{REF_LINK}>

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


def expected_normalized_cooler(handle: HiconaCooler, path: str, *, nproc: int = 4):
    f"""Create a new cooler with counts normalized by expected counts.

    Using cooltools, compute both expected count for intra chromosomal interactions
    (balanced, smoothened, not aggregated) and inter chromosomal ones (balanced).
    Then, normalize the counts in the initial cooler by dividing them for the
    corresponding expected counts. Save the results to a new cooler.

    Parameters
    ----------
    handle : HiconaCooler
        Open handle for the cooler to normalize.
    nproc : int
        Number of processes to spawn when using cooltools.

    Warning
    -------
    Due to the way matrix balancing works, some pixels will end up with count
    equal to zero and will thus be dropped. For reason, the number of output
    pixels will likely be smaller than the input one.

    Note
    ----
    Loosely based on code in the [cooltools.sandbox]<{REF_LINK}>

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
