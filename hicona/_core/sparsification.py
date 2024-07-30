"""Default functions for pixel table sparsification."""

from functools import partial

import numpy as np
import pandas as pd
import scipy as sp

from hicona._numeric import rounding


def _score_weighted(
    chunk: pd.DataFrame,
    counts_col: str,
    stats: pd.DataFrame,
    bonferroni: bool,
) -> pd.DataFrame:
    """Return the weighted sparsification scores for a chunk."""

    def _get_alpha(row: pd.Series, bonferroni: bool) -> float:
        """Compute alpha value according to Serrano et al. 2009."""

        weight, deg = row["norm_weight"], row["degree"]
        res, _ = sp.integrate.quad(lambda x: (1 - x) ** (deg - 2), 0, weight)
        alpha = 1 - (deg - 1) * res
        alpha = min(1, alpha * deg) if bonferroni else alpha

        return rounding.round_half_up(alpha, 4)

    scores: dict[str, pd.Series] = {}

    for col in ["bin1_id", "bin2_id"]:

        # Add degree and norm_weight columns to the chunk
        dataf = chunk.merge(stats, how="left", left_on=col, right_index=True)
        dataf["norm_weight"] = dataf[counts_col] / dataf.weight

        dedup = dataf[["degree", "norm_weight"]].drop_duplicates()
        dedup["alpha"] = 1.0  # .0 needed to initialize as float
        mask = dedup.degree != 1

        spar_func = partial(_get_alpha, bonferroni=bonferroni)
        dedup.loc[mask, "alpha"] = dedup.loc[mask].apply(spar_func, axis=1)

        # Merge the alpha values back into the chunk
        dataf = dataf.merge(dedup, how="left", on=["degree", "norm_weight"])
        scores[col] = dataf.alpha

    return pd.DataFrame(scores)


    chunk: pd.DataFrame,
    stats: pd.DataFrame,






def sparsify_chunk(
    chunk: pd.DataFrame, mode: str, column: str, **kwargs
) -> pd.DataFrame:
    """Return the sparsified scores for a chunk."""

    spar_functions = {"weighted": _score_weighted}  # "local_deg": _score_local_deg}
    func = spar_functions.get(mode)
    cols = ["alpha_min", "alpha_max"]

    if not func:
        raise ValueError(f"Sparsification mode must be one of {list(spar_functions)}.")

    # Compute the scores and sort them into min and max columns
    scores = func(chunk, column, **kwargs)
    scores = pd.DataFrame({cols[0]: scores.min(axis=1), cols[1]: scores.max(axis=1)})

    return scores
