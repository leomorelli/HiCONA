"""Placeholder

Placeholder
"""

from math import dist

import cooler
import numpy as np
import pandas as pd
import h5py

from .utils import round_half_up, wait_hdf5_lock


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

        table = self._fetch_chunk(lower, upper)

        self._curr_chunk += 1

        return table

    @wait_hdf5_lock
    def _fetch_chunk(self, lower, upper):
        with h5py.File(self._store_uri, mode="r") as h5_handle:
            grp = h5_handle[self._table_uri]
            table = pd.DataFrame({f: grp[f][lower:upper] for f in grp.keys()})

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
        """Placeholder"""
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

    def get_dataframe(self):
        """Placeholder"""

        return pd.concat(self.get_chunks()).reset_index(drop=True)


class RawChromTable(ChromTable):
    """Placeholder"""


class SparChromTable(ChromTable):
    """Placeholder"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._table_info = None
        self._alpha_optimal = None
        self._alpha_grid = None
        self._alpha_mod = "alpha_min"

        # TODO: Add attributes (such as filtering params)

    @property
    def table_info(self):
        """Placeholder"""
        return self._table_info

    @property
    def alpha_optimal(self):
        """Placeholder"""
        return self._alpha_optimal

    @property
    def alpha_grid(self):
        """Placeholder"""
        return self._alpha_grid

    @property
    def alpha_modality(self):
        """Placeholder"""
        modality = self._alpha_mod.split("_")[1]  # 'min'/'max'
        return modality

    @alpha_modality.setter
    def alpha_modality(self, modality: str):
        """Placeholder"""
        if modality in ("min", "max"):
            if self._alpha_mod.split("_")[1] != modality:
                self._alpha_mod = f"alpha_{modality}"
                self._alpha_grid = None
                self._alpha_optimal = None
        else:
            print("W: ignored, only 'min' and 'max' modalities are allowed.")

    def _get_alpha_pts(self, thresholds):
        """Return dataframe with filtering statistics for a threshold grid."""
        # NOTE: Assumed that alpha thresholds are in decreasing order.

        def num_nodes(table):
            """Count unique nodes in pixels-like table."""
            return len(set(table["bin1_id"]) | set(table["bin2_id"]))

        # Initialize input parameters and output container
        cur_table = self.get_dataframe()
        res_list = []

        # Define normalization values
        tot_nodes = num_nodes(cur_table)
        tot_edges = cur_table.size

        # Iterate over each alpha value
        for alpha in thresholds:
            # Filter and compute filtering statistics
            cur_table = cur_table[cur_table[self._alpha_mod] <= alpha]
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
        verbose : bool, optional
            Print progress to console. (default is True)

        Returns
        -------
        float :
            Optimal alpha value
        """

        # Initialize optimal alpha and result container
        opt_alpha = 0.5  # Middle of initial search space 0-1
        num_points = 11  # Number of points to guarantee 1 unit span
        alpha_vals = []

        for pos in range(decimals):
            if verbose:
                print(f"Optimal alpha: computing decimal {pos+1}")

            # Define the grid of alpha values to test
            step = 0.5 * 10 ** (-pos)
            grid = np.linspace(opt_alpha + step, opt_alpha - step, num_points)
            grid = [a for a in grid if 0 <= a <= 1]

            # Compute the new statistics, then update optimal alpha
            new_pts = self._get_alpha_pts(grid)
            opt_alpha = new_pts["alpha"].iloc[new_pts["eu_dist"].idxmin()]
            alpha_vals.append(new_pts)

        # Store results
        alpha_vals = pd.concat(alpha_vals).reset_index()
        opt_alpha = round_half_up(opt_alpha, decimals)  # "2.129999" -> "2.13"
        self._alpha_grid = alpha_vals
        self._alpha_optimal = opt_alpha

        if verbose:
            print(f"Optimal alpha: {opt_alpha}")

        return opt_alpha

    def get_dataframe(self, alpha: str | float = None) -> pd.DataFrame:
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
            if not self._alpha_optimal:
                raise ValueError("W: ignored, compute optimal alpha first.")
            alpha = self._alpha_optimal

        if alpha is not None:
            chunks = [c[c[self._alpha_mod] < alpha] for c in self.get_chunks()]
        else:
            chunks = [c for c in self.get_chunks()]

        return pd.concat(chunks).reset_index(drop=True)

    def bin_annotation_pairs(self, annot: str, bins, alpha: str | float = None):
        """Placeholder"""

        ann_df = cooler.annotate(self.get_dataframe(), bins)

        # Sort annotations alphabetically to make a triagular matrix later on
        ann_df[f"{annot}1"], ann_df[f"{annot}2"] = np.where(
            ann_df[f"{annot}1"] < ann_df[f"{annot}2"],
            (ann_df[f"{annot}1"], ann_df[f"{annot}2"]),
            (ann_df[f"{annot}2"], ann_df[f"{annot}1"]),
        )

        table = ann_df.groupby([f"{annot}1", f"{annot}2"])["count"].count()
        table = table.unstack().T
        table = table / len(self.get_dataframe())

        print(np.nansum(table.to_numpy()))
        assert np.nansum(table.to_numpy()) == 1


"""
    
    annotated = cooler.annotate(pixel_tab, handle.bins()[:])


annotated["HMM_annot1"], annotated["HMM_annot2"] = np.where(
    annotated["HMM_annot1"] < annotated["HMM_annot2"],
    (annotated["HMM_annot1"], annotated["HMM_annot2"]),
    (annotated["HMM_annot2"], annotated["HMM_annot1"]),
)

comparison = annotated.groupby(["HMM_annot1", "HMM_annot2"])["count"].count()
comparison = comparison.unstack().T
comparison = comparison / len(pixel_tab)

print(np.nansum(comparison.to_numpy()))
assert np.nansum(comparison.to_numpy()) == 1

sns.heatmap(comparison, annot=True, robust=True)
plt.show()
"""
