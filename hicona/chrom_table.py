"""Placeholder

Placeholder
"""

from collections import deque
from math import dist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import h5py

from .utils import round_half_up


class TableChunksIterator:
    """Return table chunks as pandas DataFrames."""

    def __init__(self, store_uri, table_uri, idx_bounds, chunk_size):
        self._store_uri = store_uri
        self._table_uri = table_uri
        self._idx_bounds = idx_bounds
        self._chunk_size = chunk_size

        # Set iteration properties
        self._curr_chunk = 0
        self._num_chunks = (idx_bounds[1] - idx_bounds[0]) // chunk_size
        if (idx_bounds[1] - idx_bounds[0]) % chunk_size != 0:
            self._num_chunks += 1

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr_chunk >= self._num_chunks:
            raise StopIteration

        # Fix boundaries of the region to fetch
        lower = self._idx_bounds[0] + self._chunk_size * self._curr_chunk
        upper = lower + self._chunk_size
        if upper > self._idx_bounds[1]:
            upper = self._idx_bounds[1]

        with h5py.File(self._store_uri, mode="r") as h5_handle:
            grp = h5_handle[self._table_uri]
            table = pd.DataFrame({f: grp[f][lower:upper] for f in grp.keys()})

        self._curr_chunk += 1

        return table


class ChromTablesIterator:
    """Iterator object of chromosome-level tables and respective information.

    For each table group specified in a list of URI strings, return a
    :py:class:`ChromTable` object whose data attribute corresponds to all
    tables in the group, while the preprocessing_params contains all the
    parameters used for processing plus the chromosome id.

    Parameters
    ----------
    store : str
        Path to the cool/mcool file.
    root : str
        URI string to resolution of interest.
    uris : list
        List of URI strings to the table groups of interest.
    """

    def __init__(self, store, uris):
        self._store = store
        self._uri_list = uris

        self._uri_index = 0
        self._max_uri = len(uris)

    def __len__(self):
        return self._max_uri

    def __iter__(self):
        return self

    def __next__(self):
        if self._uri_index >= self._max_uri:
            raise StopIteration

        curr_uri = self._uri_list[self._uri_index]
        self._uri_index += 1

        with h5py.File(self._store, mode="r") as h5_handle:
            table_grp = h5_handle[curr_uri]

        return SparChromTable(self._store, curr_uri)


class ChromTable:
    """Placeholder"""

    def __init__(
        self,
        store_uri: str,
        table_uri: str,
        idx_bounds: tuple[int] = None,
        chunk_size: int = 1_000_000,
    ):
        self._store_uri = store_uri
        self._table_uri = table_uri
        self._idx_bounds = idx_bounds
        self._chunk_size = chunk_size

        # Fetch bin size from the main file (needed for filtering)
        with h5py.File(self._store_uri, mode="r") as h5_handle:
            grp = h5_handle[self._table_uri]
            self._bin_size = grp.attrs.get("bin-size")

            if self._idx_bounds is None:
                self._idx_bounds = (0, len(grp["bin1_id"]))

        # TODO: Create defaults object maybe

    @property
    def store_uri(self):
        """Uri string to the cooler of interest (includes resolution)."""
        return self._store_uri

    @property
    def table_uri(self):
        """Uri string to the table of interest from the main cooler."""
        return self._table_uri

    @property
    def chunk_size(self):
        """Size of the chunks to retrieve during iteration."""
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, size: int):
        if isinstance(size, int):
            self._chunk_size = size
        else:
            print("W: invalid chunk size provided, value not updated.")

    @property
    def bin_size(self):
        return self._bin_size

    def get_chunks(self) -> TableChunksIterator:
        """Returns an iterator of table chunks (as pandas DataFrames)."""

        chunks = TableChunksIterator(
            self._store_uri,
            self._table_uri,
            self._idx_bounds,
            self._chunk_size,
        )

        return chunks


class ChromTableOld:
    """Class used to handle preprocessed chromosome level table.

    Class implementing methods for further manipulation of already
    preprocessed chromosome level tables. It allows to prepare the tables
    for network conversion (for instance by filtering pixels based on
    sparsification results). Class mostly meant for instantiation through
    a :py:class:`ChromTablesIterator` instance.

    Parameters
    ----------
    pix_table : :py:class:`DataFrame`
        Pixels-like table (e.i. "bin1_id", "bin2_id", "count" columns must
        be present).
    params_info : dict
        Dictionary containing information about table preprocessing.
    """

    def __init__(self, pix_table: pd.DataFrame, params_info: dict):
        self._data = pix_table
        self._params_info = params_info
        self._alpha_column = "alpha_min"
        self._optim_alpha = None
        self._alphas_grid = None

    @property
    def data(self) -> pd.DataFrame:
        """:py:class:`DataFrame` of pixels."""
        return self._data

    @property
    def params_info(self) -> dict:
        """Dictionary of parameters used during preprocessing."""
        return self._params_info

    @property
    def optim_alpha(self) -> float | None:
        """Optimal alpha to filter the pixels (if computed, else None)."""
        return self._optim_alpha

    @property
    def alpha_column(self) -> str:
        """Name of the column used as alpha scores."""
        return self._alpha_column

    @alpha_column.setter
    def alpha_column(self, col_name) -> None:
        """Set the column to use as alpha scores."""
        if col_name in ("alpha_min", "alpha_max"):
            self._alpha_column = col_name
            # Reset previous optimal alpha and grid if existent.
            self._optim_alpha = None
            self._alphas_grid = None
        else:
            print("W: ignored, only 'alpha_min' and 'alpha_max' are allowed.")

    def _get_alpha_pts(self, thresholds):
        """Return dataframe with filtering statistics for a threshold grid."""
        # NOTE: Assumed that alpha thresholds are in decreasing order.

        def num_nodes(table):
            """Count unique nodes in pixels-like table."""
            return len(set(table["bin1_id"]) | set(table["bin2_id"]))

        # Initialize input parameters and output container
        alpha_thr = deque(thresholds)
        cur_table = self._data.copy()
        res_list = []

        # Define normalization values
        tot_nodes = num_nodes(cur_table)
        tot_edges = cur_table.size

        # Iterate over each alpha value
        while alpha_thr:
            alpha = alpha_thr.popleft()

            # Filter and compute filtering statistics
            cur_table = cur_table[cur_table[self._alpha_column] <= alpha]
            cur_nodes = num_nodes(cur_table) / tot_nodes
            cur_edges = cur_table.size / tot_edges

            # Store results as matrix row
            res_list.append(
                {
                    "alpha": alpha,
                    "nodes_f": cur_nodes,
                    "edges_f": cur_edges,
                    "eu_dist": dist((1, 0), (cur_nodes, cur_edges)),
                }
            )

        return pd.DataFrame(res_list)

    def compute_opt_alpha(
        self,
        decimals: int = 3,
        num_pts: int = 11,
        verbose: bool = True,
    ) -> float:
        """Return the optimal alpha value for filtering the pixel table.

        Compute the optimal alpha value for by iterating these steps:

        - define ``num_pts`` alpha values centered around the current optimal
          and spaced in order to cover 1 unit of previous decimal position.
        - for each alpha value, filter the table and compute the remaining
          fraction of edges and nodes.
        - for each alpha compute the Euclidean distance from point ``(1, 0)``
          in the space ``x: fraction of nodes``, ``y: fraction of edges``.
        - set as new optimal alpha the one minimizing the Euclidean distance.
        - move to the next decimal position and repeat the procedure.

        Parameters
        ----------
        decimals : int, optional
            Number of decimal positions to compute for the alpha value. Must
            be at least 1. (default is 3)
        num_pts : int, optional
            Number of alpha values to test at each decimal position. Must be
            at least 3. (default is 11)
        verbose : bool, optional
            Print progress to console. (default is True)

        Returns
        -------
        float :
            Optimal alpha value
        """

        # TODO: If num_pts is removed, add check of pre-existing alphas
        # TODO: Maybe add recompute parameter but not the most elegant
        # TODO: Could add check that minimum is not grid extrema

        # Initialize optimal alpha and result container
        opt_alpha = 0.5  # Middle of initial search space 0-1
        alpha_vals = []

        for pos in range(decimals):
            if verbose:
                print(f"Optimal alpha: computing decimal {pos+1}")

            # Define the grid of alpha values to test
            step = 0.5 * 10 ** (-pos)
            grid = np.linspace(opt_alpha + step, opt_alpha - step, num_pts)
            grid = [a for a in grid if 0 <= a <= 1]

            # Compute the new statistics, then update optimal alpha
            new_pts = self._get_alpha_pts(grid)
            opt_alpha = new_pts["alpha"].iloc[new_pts["eu_dist"].idxmin()]
            alpha_vals.append(new_pts)

        # Store results
        alpha_vals = pd.concat(alpha_vals).reset_index()
        opt_alpha = round_half_up(opt_alpha, decimals)
        self._alphas_grid = alpha_vals
        self._optim_alpha = opt_alpha

        if verbose:
            print(f"Optimal alpha: {opt_alpha}")

        return opt_alpha

    def filter_alpha(self, alpha: str | float = "optimal") -> pd.DataFrame:
        """Return the pixel table filtered according to some alpha value.

        Return a :py:class:`DataFrame` where only the pixels having a
        sparsification alpha value below the provided threshold are kept.

        Parameters
        ----------
        alpha : str or float, optional
            Keeping only the pixels with alpha smaller than this threshold.
            If "optimal", use the previously computed optimal alpha value.
            (default is "optimal")

        Returns
        -------
        :py:class:`DataFrame` :
            Dataframe of filtered pixels.
        """

        if alpha == "optimal":
            if not self._optim_alpha:
                msg = "Cannot filter by optimal alpha before computing it."
                raise UnboundLocalError(msg)
            alpha = self._optim_alpha

        filt_df = self._data.copy()
        filt_df = filt_df[filt_df[self._alpha_column] < alpha]

        return filt_df
