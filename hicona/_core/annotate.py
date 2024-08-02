"""Mimick cooler's annotate but allowing for polars dfs."""

from typing import overload

import pandas as pd
import polars as pl

from hicona._dtypes import GenericDf


@overload
def annotate(pixels: pl.DataFrame, bins: GenericDf) -> pl.DataFrame: ...
@overload
def annotate(pixels: pd.DataFrame, bins: GenericDf) -> pd.DataFrame: ...


def annotate(pixels: GenericDf, bins: GenericDf) -> GenericDf:
    """Add bin annotations to a data frame of pixels.

    Perform a left join of the pixels with the bins data frame, adding individual
    bin information to each pixel. Annotation columns are suffixed with '1' and '2'
    depending on the bin they refer to.

    Parameters
    ----------
    pixels : polars.DataFrame or pandas.DataFrame
        A data frame of pixels with the columns 'bin1_id' and 'bin2_id' (any
        additional columns can be present and will be retained).
    bins : polars.DataFrame or pandas.DataFrame
        A data frame of bins with the columns 'bin_id' and any additional columns
        to be added to the pixels.

    Returns
    -------
    polars.DataFrame or pandas.DataFrame
        A data frame of pixels with the bin information added.

    Notes
    -----
    This function is similar to the ``cooler.annotate`` function, but allows for
    polars data frames as inputs and outputs a dataframe of the same type as the
    input pixels dataframe. Unlike ``cooler.annotate``, there is no guarantee of
    column order.

    """

    # Convert to polars DataFrames if necessary
    pixels = pixels if isinstance(pixels, pl.DataFrame) else pl.DataFrame(pixels)
    if isinstance(bins, pd.DataFrame):
        bins = (
            pl.DataFrame(bins)
            .with_row_index("bin_id")
            .with_columns(pl.col("bin_id").cast(pl.Int64))
        )

    # Annotate pixels with bin information
    annotated = (
        pixels.join(bins, left_on="bin1_id", right_on="bin_id", how="left")
        .join(bins, left_on="bin2_id", right_on="bin_id", how="left", suffix="2")
        .rename(lambda c: c + "1" if c in bins.columns else c)
        .select(pl.all().exclude("bin_id1", "bin_id2"))
    )

    return annotated
