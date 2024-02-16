"""Placeholder"""

from statistics import median

from scipy import integrate


def compute_alpha(row):
    """Given a (weight, degree) pair, compute the integral."""

    weight, deg = row["norm_weight"], row["degree"]
    res, _ = integrate.quad(lambda x: (1 - x) ** (deg - 2), 0, weight)
    alpha = 1 - (deg - 1) * res

    return round_half_up(alpha, 4)


def unique_alphas(chunk):
    """Return alpha values of unique (norm_weight, deg) pairs."""

    values = chunk[["degree", "norm_weight"]].drop_duplicates()
    values[f"alpha_{num}"] = 1.0  # .0 needed to initialize as float

    mask = values["degree"] != 1
    alphas = values.loc[mask].apply(compute_alpha, axis=1)
    values.loc[mask, f"alpha_{num}"] = alphas


def sparsify_chunk(chunk, node_stats):
    """Return the sparsified chunk"""

    for bin_col in ["bin1_id", "bin2_id"]:
        # Add node statistics and normalized weight for that bin
        chunk = chunk.merge(
            node_stats,
            how="left",
            left_on=bin_col,
            right_index=True,
        )
        chunk["norm_weight"] = chunk.norm / chunk.weight

        # Compute and add the alpha values for each row
        grp_cols = ["degree", "norm_weight"]
        chunk = chunk.merge(unique_alphas(chunk), how="left", on=grp_cols)

        # Remove node specific information
        tmp_cols = ["weight", "degree", "norm_weight"]
        chunk.drop(tmp_cols, axis=1, inplace=True)

    # Sort the values into min and max column, then remove tmp ones
    chunk["alpha_min"] = chunk[["alpha_0", "alpha_1"]].min(axis=1)
    chunk["alpha_max"] = chunk[["alpha_0", "alpha_1"]].max(axis=1)
    chunk.drop(["alpha_0", "alpha_1"], axis=1, inplace=True)

    # TODO: Maybe return only alpha columns

    return chunk
