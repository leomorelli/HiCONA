"""Default functions for pixel table sparsification."""

import os
import pathlib

import scipy as sp
import polars as pl
import numpy as np

from hicona._numeric import rounding


def _score_weighted(
    chunk: pl.DataFrame,
    counts_col: str,
    stats: pl.DataFrame,
    bonferroni: bool,
) -> pl.DataFrame:
    """Return the weighted sparsification scores for a chunk."""

    def _get_alpha(weight: float, degree: int) -> float:
        """Compute alpha value according to Serrano et al. 2009."""

        res, _ = sp.integrate.quad(lambda x: (1 - x) ** (degree - 2), 0, weight)
        alpha = 1 - (degree - 1) * res
        return rounding.round_half_up(alpha, 4)

    scores: pl.DataFrame = pl.DataFrame()

    for col in ["bin1_id", "bin2_id"]:

        # Add degree and norm_weight columns to the chunk
        dataf = chunk.join(
            stats, how="left", left_on=col, right_on="bin_id"
        ).with_columns((pl.col(counts_col) / pl.col("weight")).alias("norm_weight"))

        # Compute alpha values for each degree and norm_weight
        alphas = (
            dataf.select(["degree", "norm_weight"])
            .filter(pl.col("degree") != 1)
            .unique()
            .with_columns(
                pl.struct(["degree", "norm_weight"])
                .map_elements(
                    lambda x: _get_alpha(x["norm_weight"], x["degree"]),
                    return_dtype=pl.Float64,
                )
                .alias("alpha")
            )
        )

        # TODO: Add back bonferroni correction (maybe)
        if bonferroni:
            raise NotImplementedError("Bonferroni correction not yet implemented.")

        dataf = dataf.join(
            alphas, how="left", on=["degree", "norm_weight"]
        ).with_columns(pl.col("alpha").fill_null(1.0))

        scores = scores.with_columns(dataf["alpha"].alias(col))

    scores = scores.with_columns(
        pl.min_horizontal("bin1_id", "bin2_id").alias("score_min"),
        pl.max_horizontal("bin1_id", "bin2_id").alias("score_max"),
    ).select(["score_min", "score_max"])

    return scores


def _score_local_deg(
    chunk: pl.DataFrame,
    ranking: pathlib.Path,
    degrees: pl.DataFrame,
    merge_col: str,
) -> pl.DataFrame:
    """Return the local degree sparsification scores for a chunk."""

    # Add degrees to the chunk as well as empty score columns
    for col in [1, 2]:
        chunk = chunk.join(
            degrees.select(pl.all().name.prefix(f"bin{col}_")),
            how="left",
            left_on=f"bin{col}_id",
            right_on=f"bin{col}_bin_id",  # Not the prettiest but simpler
        ).with_columns(pl.lit(0).alias(f"bin{col}_rank"))

    # Limits of the columns of each chunk, used to choose when the merge is necessary
    # since merging is expensive and with sorted dataframes we can avoid it partially.
    # TODO: Make more readable and probably move to a separate function
    # TODO: Could also probably be improved by taking first and last elements.
    limits: dict[str, int] = {
        "bin1_lower": int(chunk["bin1_id"].min()),  # type: ignore
        "bin1_upper": int(chunk["bin1_id"].max()),  # type: ignore
        "bin2_lower": int(chunk["bin2_id"].min()),  # type: ignore
        "bin2_upper": int(chunk["bin2_id"].max()),  # type: ignore
    }

    # Annotate with all ranking chunks that overlap with the pixel chunk
    for rank_path in os.listdir(ranking):

        # Only use single-file parquet files (avoid intermediate chunks)
        if not rank_path.endswith(".parquet"):
            continue

        # Retrieve min and max bin id in the chunk
        rank_min, rank_max = map(int, rank_path.split(".")[0].split("-"))

        # If there is no overlap between the pixel chunk and the ranking chunk, skip
        if rank_max < limits["bin1_lower"] or rank_min > limits["bin2_upper"]:
            continue

        # Do it on both bin1 and bin2 columns
        # TODO: make this more readable
        for bin_col, deg_col in [(1, 2), (2, 1)]:

            # If there is no overlap between the column and the ranking chunk, skip
            if (
                rank_max < limits[f"bin{bin_col}_lower"]
                or rank_min > limits[f"bin{bin_col}_upper"]
            ):
                continue

            # Update the bin rank column with the ranking information if present
            chunk = (
                chunk.join(
                    pl.read_parquet(ranking / rank_path),
                    how="left",
                    left_on=[f"bin{bin_col}_id", f"bin{deg_col}_{merge_col}"],
                    right_on=["bin_id", merge_col],
                )
                .with_columns(
                    pl.when(pl.col("rank") > 0)
                    .then(pl.col("rank"))
                    .otherwise(pl.col(f"bin{deg_col}_rank"))
                    .alias(f"bin{deg_col}_rank")
                )
                .drop("rank")
            )

    # Actually compute the scores
    for bin_col, deg_col in [(1, 2), (2, 1)]:
        chunk = chunk.with_columns(
            pl.when(pl.col(f"bin{deg_col}_degree") > 1)
            .then(
                pl.lit(1)
                - np.log(pl.col(f"bin{bin_col}_rank"))
                / np.log(pl.col(f"bin{deg_col}_degree"))
            )
            .otherwise(1)
            .alias(f"bin{bin_col}_score")
        )

    # Sort the scores into min and max columns
    scores = chunk.with_columns(
        pl.min_horizontal("bin1_score", "bin2_score").alias("score_min"),
        pl.max_horizontal("bin1_score", "bin2_score").alias("score_max"),
    ).select(["score_min", "score_max"])

    return scores


def sparsify_chunk(chunk: pl.DataFrame, mode: str, **kwargs) -> pl.DataFrame:
    """Return the sparsified scores for a chunk."""

    spar_functions = {
        "weighted": _score_weighted,
        "local_deg": _score_local_deg,
    }
    func = spar_functions.get(mode)

    if not func:
        raise ValueError(f"Sparsification mode must be one of {list(spar_functions)}.")

    # Compute the scores and sort them into min and max columns
    scores = func(chunk, **kwargs)

    # TODO: remove rename once migrated from "alpha" to "score".
    scores = scores.rename({"score_min": "alpha_min", "score_max": "alpha_max"})

    return pl.concat([chunk.drop("alpha_min", "alpha_max"), scores], how="horizontal")
