"""Default functions for pixel table sparsification."""

import logging
import time

import pandas as pd
import scipy as sp

from hicona._core import base_table
from hicona._ops import chunked, hdf5
from hicona._numeric import rounding
from hicona._table import HiconaTable


def _compute_alpha(row: pd.Series) -> float:
    """Compute alpha value according to Serrano et al. 2009."""

    weight, deg = row["norm_weight"], row["degree"]
    res, _ = sp.integrate.quad(lambda x: (1 - x) ** (deg - 2), 0, weight)
    alpha = 1 - (deg - 1) * res

    return rounding.round_half_up(alpha, 4)


def _get_alphas(chunk: pd.DataFrame, stats: pd.DataFrame, col: str) -> pd.Series:
    """Return the alpha values for the given chunk."""

    # Add degree and norm_weight columns to the chunk
    dataf = chunk.merge(stats, how="left", left_on=col, right_index=True)
    dataf["norm_weight"] = dataf.norm / dataf.weight

    # Compute the alpha values
    dedup = dataf[["degree", "norm_weight"]].drop_duplicates()
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


class TableProcessor:
    """Process a table according to the given operations."""

    def __init__(self, table: base_table.Table, log_level: int = logging.INFO):
        self._table = table
        self._log_level = log_level

    def _apply_functions(self):
        """ "Apply all filtering and normalization functions to the table."""

        for op in self._table.flow.get_partials():
            logging.info("Starting to apply: %s", op.func.__name__)

            tab_size = hdf5.write_table(self._table.uris, op(self._table))
            hdf5.resize_table(self._table.uris, tab_size)
            self._table.reset_index()

            logging.info("Finished applying: %s", op.func.__name__)
            logging.info("Table size: %s", tab_size)

    def _spar_table(self):
        """Sparsify the chunks and save the results."""

        logging.info("Starting to sparsify the table.")

        node_stats = chunked.get_node_stats(self._table.chunks(), "norm")
        tab_size = 0

        # NOTE: implemented this way to simplify breaking into parallel later
        for i, chunk in enumerate(self._table.chunks()):

            logging.info("Starting to sparsify chunk %s.", i)
            start = time.time()

            spar_chunk = sparsify_chunk(chunk, node_stats)
            hdf5.write_chunk(self._table.uris, spar_chunk, tab_size)

            end = time.time()
            logging.info("Finished sparsifying chunk %s.", i)
            logging.info("Took %s seconds.", end - start)

            tab_size += len(spar_chunk)

    def create_table(self) -> HiconaTable:
        """Begin actual table processing."""

        logging.basicConfig(level=self._log_level)
        logging.info("Starting table creation.")

        start_time = time.time()

        self._apply_functions()
        self._spar_table()

        end_time = time.time()
        logging.info("Table creation took %s seconds.", end_time - start_time)

        return HiconaTable(self._table.uris)
