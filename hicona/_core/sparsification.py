"""Default functions for pixel table sparsification."""

import scipy as sp
import polars as pl

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
                .map_elements(lambda x: _get_alpha(x["norm_weight"], x["degree"]))
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


# def _score_local_deg(
#     chunk: pd.DataFrame,
#     counts_col: str,
#     stats: pd.DataFrame,
# ) -> pd.DataFrame:

#     def compute_scores(df):

#         data = df.copy()
#         data["score"] = 1.0

#         index = data["degree"] > 1
#         redux = data.loc[index, :].copy()
#         redux["score"] = 1 - (np.log(redux["rank"]) / np.log(redux["degree"]))
#         data.loc[index, "score"] = redux["score"]

#         return data

#     scores: dict[str, pd.Series] = {}
#     scores_df = compute_scores(stats)

#     for col in ["bin1_id", "bin2_id"]:

#         merge = chunk.merge(
#             scores_df,
#             how="left",
#             left_on=[col, counts_col],
#             right_on=["bin_id", "count"],
#         )

#         scores[col] = merge["score"]

#     return pd.DataFrame(scores)


def sparsify_chunk(
    chunk: pl.DataFrame, mode: str, column: str, **kwargs
) -> pl.DataFrame:
    """Return the sparsified scores for a chunk."""

    spar_functions = {"weighted": _score_weighted}  # "local_deg": _score_local_deg}
    func = spar_functions.get(mode)
    cols = ["alpha_min", "alpha_max"]

    if not func:
        raise ValueError(f"Sparsification mode must be one of {list(spar_functions)}.")

    # Compute the scores and sort them into min and max columns
    scores = func(chunk, column, **kwargs)
    scores = pl.DataFrame({cols[0]: scores.min(axis=1), cols[1]: scores.max(axis=1)})

    return scores
