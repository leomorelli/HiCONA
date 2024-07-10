"""Default functions for pixel table sparsification."""

from functools import partial

import pandas as pd
import scipy as sp

from hicona._numeric import rounding


def _compute_alpha(row: pd.Series, bonferroni: bool) -> float:
    """Compute alpha value according to Serrano et al. 2009."""

    weight, deg = row["norm_weight"], row["degree"]
    res, _ = sp.integrate.quad(lambda x: (1 - x) ** (deg - 2), 0, weight)
    alpha = 1 - (deg - 1) * res

    if bonferroni:
        alpha = min(1, alpha * deg)

    return rounding.round_half_up(alpha, 4)


def _get_alphas(
    chunk: pd.DataFrame,
    stats: pd.DataFrame,
    col: str,
    bonferroni: bool,
) -> pd.Series:
    """Return the alpha values for the given chunk."""

    # Add degree and norm_weight columns to the chunk
    dataf = chunk.merge(stats, how="left", left_on=col, right_index=True)
    dataf["norm_weight"] = dataf.norm / dataf.weight

    # Compute the alpha values
    dedup = dataf[["degree", "norm_weight"]].drop_duplicates()
    dedup["alpha"] = 1.0  # .0 needed to initialize as float
    mask = dedup.degree != 1
    spar_func = partial(_compute_alpha, bonferroni=bonferroni)
    dedup.loc[mask, "alpha"] = dedup.loc[mask].apply(spar_func, axis=1)

    # Merge the alpha values back into the chunk
    dataf = dataf.merge(dedup, how="left", on=["degree", "norm_weight"])

    return dataf.alpha


def sparsify_chunk(
    chunk: pd.DataFrame,
    node_stats: pd.DataFrame,
    bonferroni: bool,
) -> pd.DataFrame:
    """Return the sparsified chunk"""

    alphas = {
        i: _get_alphas(chunk, node_stats, f"bin{i}_id", bonferroni) for i in range(1, 3)
    }
    alphas = pd.DataFrame(alphas)

    # Sort the values into min and max column, then remove tmp ones
    res = {"alpha_min": alphas.min(axis=1), "alpha_max": alphas.max(axis=1)}
    return pd.DataFrame(res)
