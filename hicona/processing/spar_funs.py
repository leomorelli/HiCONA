"""Default functions for pixel table sparsification."""

import pandas as pd
from scipy import integrate

from ..utils.numeric import round_half_up


def _compute_alpha(row: pd.Series) -> float:
    """Given a (weight, degree) pair, compute the integral."""

    weight, deg = row["norm_weight"], row["degree"]
    res, _ = integrate.quad(lambda x: (1 - x) ** (deg - 2), 0, weight)
    alpha = 1 - (deg - 1) * res

    return round_half_up(alpha, 4)


def _get_alphas(chunk: pd.DataFrame, stats: pd.DataFrame, col: str) -> pd.Series:
    """Return the alpha values for the given chunk."""

    # Add degree and norm_weight columns to the chunk
    dataf = chunk.merge(stats, how="left", left_on=col, right_index=True)
    dataf["norm_weight"] = dataf.norm / dataf.weight

    # Compute the alpha values
    dedup = dataf[["degree", "norm_weight"]].drop_duplicates()

    # print("NEW CHUNK")
    # print("Chunk size pre-dedup:", len(dataf))
    # print("Chunk size post-dedup:", len(dedup))
    # print("Chunk size reduction:", len(dataf) - len(dedup))

    dedup["alpha"] = 1.0  # .0 needed to initialize as float
    mask = dedup.degree != 1
    dedup.loc[mask, "alpha"] = dedup.loc[mask].apply(_compute_alpha, axis=1)

    # Merge the alpha values back into the chunk
    dataf = dataf.merge(dedup, how="left", on=["degree", "norm_weight"])

    return dataf.alpha


def sparsify_chunk(chunk: pd.DataFrame, node_stats: pd.DataFrame) -> pd.DataFrame:
    """Return the sparsified chunk"""

    alphas = {i: _get_alphas(chunk, node_stats, f"bin{i}_id") for i in range(1, 3)}
    alphas = pd.DataFrame(alphas)

    # Sort the values into min and max column, then remove tmp ones
    res = {"alpha_min": alphas.min(axis=1), "alpha_max": alphas.max(axis=1)}
    return pd.DataFrame(res)
