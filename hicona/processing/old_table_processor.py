"""Placeholder"""

from collections.abc import Iterable
from statistics import median

from numpy import log2
import pandas as pd
from scipy import integrate

from .hicona_table import HiconaTable
from .utils.numeric import round_half_up


class TableProcessor:
    """Placeholder"""

    def __init__(self, store, root, table_path, all_kwargs):
        quant = queries.pop("quantile_thr")

        self._table = table
        self._quant = quant
        self._queries = queries
        self._verbose = verbose

        # Computed once during the first iteration
        self._table_size = None
        self._norm_curve = None
        self._quant_numb = None
        self._node_stats = None

    def _compute_norm_curve(self):
        """Compute curve for genomic distance normalization."""

        curve = pd.Series()
        for chunk in self._get_filtered_chunks():
            chunk["bin_diff"] = chunk["bin2_id"] - chunk["bin1_id"]
            vals = chunk.groupby("bin_diff")["count"].apply(list)
            curve = curve.combine(vals, lambda x, y: x + y, fill_value=[])

        self._norm_curve = curve.apply(median).rename("dist_norm")

    def _get_normalized_chunks(self):
        """Iterator of table chunks normalized for genomic distance."""

        if self._norm_curve is None:
            self._compute_norm_curve()

        for chunk in self._get_filtered_chunks():
            # Compute expected ratios
            chunk["bin_diff"] = chunk["bin2_id"] - chunk["bin1_id"]
            chunk = chunk.join(self._norm_curve, on="bin_diff")
            chunk["exp_ratio"] = log2(chunk["count"] / chunk["dist_norm"] + 1)
            chunk.drop(["bin_diff", "dist_norm"], axis=1, inplace=True)

            yield chunk

    def _compute_quant_numb(self):
        """Define the expected ratio cutoff to use for quantile filtering."""

        curve = pd.Series()
        for chunk in self._get_normalized_chunks():
            vals = chunk.groupby("exp_ratio")["count"].count()
            curve = curve.combine(vals, lambda x, y: x + y, fill_value=0)

        threshold = curve.sum() * self._quant
        threshold = curve.cumsum()[curve.cumsum() > threshold].index[0]

        self._quant_numb = threshold

    def _get_quantile_chunks(self):
        """Filter chromsome pixels and store them in the table."""

        if not self._quant_numb:
            self._compute_quant_numb()

        for chunk in self._get_normalized_chunks():
            yield chunk[chunk["exp_ratio"] > self._quant_numb]

    def _compute_node_stats(self):
        """Compute sum of weights and degree for each node/bin."""

        # Initialize empty containers
        weights = pd.Series()
        degrees = pd.Series()

        for chunk in self._get_quantile_chunks():
            for bin_col in ["bin1_id", "bin2_id"]:
                # Compute metrics on chunk
                grouped = chunk[[bin_col, "exp_ratio"]].groupby(bin_col)
                chunk_weights = grouped.sum()["exp_ratio"]
                chunk_degrees = grouped.count()["exp_ratio"]

                # Increase counters
                weights = weights.add(chunk_weights, fill_value=0)
                degrees = degrees.add(chunk_degrees, fill_value=0)

        node_stats = {"weight": weights, "degree": degrees}
        self._node_stats = pd.DataFrame(node_stats)

    def _get_sparsified_chunks(self):
        """Compute alpha value as per Serrano et al. 2009."""

        def compute_alpha(row):
            """Given a (weight, degree) pair, compute the integral."""

            weight, deg = row["norm_weight"], row["degree"]
            res, _ = integrate.quad(lambda x: (1 - x) ** (deg - 2), 0, weight)
            alpha = 1 - (deg - 1) * res

            return round_half_up(alpha, 4)

        def unique_alphas(dataf):
            """Return alpha values of unique (norm_weight, deg) pairs."""

            values = dataf[["degree", "norm_weight"]].drop_duplicates()
            values[f"alpha_{num}"] = 1.0  # .0 needed to initialize as float
            mask = values["degree"] != 1
            alphas = values.loc[mask].apply(compute_alpha, axis=1)
            values.loc[mask, f"alpha_{num}"] = alphas

            return values

        if not self._node_stats:
            self._compute_node_stats()

        # For each chunk of the chromosome-level pixel table
        for chunk in self._get_quantile_chunks():
            # For both bins composing the pixel
            for num, bin_col in enumerate(["bin1_id", "bin2_id"]):
                # Add node statistics and normalized weight for that bin
                chunk = chunk.merge(
                    self._node_stats,
                    how="left",
                    left_on=bin_col,
                    right_index=True,
                )
                chunk["norm_weight"] = chunk["exp_ratio"] / chunk["weight"]

                # Compute and add the alpha values for each row
                chunk = chunk.merge(
                    unique_alphas(chunk),
                    how="left",
                    on=["degree", "norm_weight"],
                )

                # Remove node specific information
                tmp_cols = ["weight", "degree", "norm_weight"]
                chunk.drop(tmp_cols, axis=1, inplace=True)

            # Sort the values into min and max column, then remove tmp ones
            chunk["alpha_min"] = chunk[["alpha_0", "alpha_1"]].min(axis=1)
            chunk["alpha_max"] = chunk[["alpha_0", "alpha_1"]].max(axis=1)
            chunk.drop(["alpha_0", "alpha_1"], axis=1, inplace=True)

            yield chunk

    def get_processed_chunks(self) -> Iterable[pd.DataFrame]:
        """Placeholder"""

        return self._get_sparsified_chunks()


def get_norm_params(norm_name: str) -> dict:
    """Return a dict with the default parameter values for a normalization.

    # TODO: finish
    """

    def get_queries(binsize, upper_idx, dist_thr, count_thr, quant_thr):
        """Get a dictionary containing the strings to use as queries."""

        queries = {}

        # QUERY: remove pixels with bins outside of the interval
        queries["out_interval"] = f"bin2_id < {upper_idx}"

        # QUERY: remove self-looping pixels
        queries["self_looping"] = "bin1_id != bin2_id"

        # QUERY: remove pixels above maximal genomic distance
        max_diff = -(-dist_thr // binsize)
        queries["genomic_dist"] = f"bin2_id - bin1_id < {max_diff}"

        # QUERY: remove pixels with raw counts below a certain theshold
        if count_thr > 0:
            queries["below_counts"] = f"count > {count_thr}"

        # QUERY: remove a quantile of pixels from the processed table
        if quant_thr > 0:
            queries["quantile_thr"] = quant_thr

        return queries

    pass
