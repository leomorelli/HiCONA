"""Placeholder

Placeholder
"""

from collections.abc import Iterable
from math import dist

import cooler
import numpy as np
import pandas as pd

import hicona.hicona_cooler as hicooler  # For circular import
from .uris import Uris
from .utils.numeric import round_half_up
from .utils.hdf5_ops import fetch_chunk, get_table_size
from .processing.processing_flow import ProcessingFlow


class ChunksIterator:
    """Iterator of table chunks as pandas DataFrames."""

    def __init__(
        self,
        uris: Uris,
        chunk_size: int,
        intervals: list[tuple[int, int]],
        columns: Iterable[str] | None = None,
        annotated: bool = False,
    ):
        self._uris = uris
        self._chunk_size = chunk_size
        self._intervals = intervals
        self._columns = columns
        self._annotated = annotated
        self._bins = pd.DataFrame()

        if self._annotated:
            cool = hicooler.HiconaCooler(self._uris.cooler_uri())
            self._bins = cool.bare_bins()

    def __iter__(self):
        return self

    def __next__(self) -> pd.DataFrame:
        # No more pixels available to iterate
        if not self._intervals:
            raise StopIteration

        out_interv = []  # Pixel intervals to include in the chunk
        still_miss = self._chunk_size  # Pixels missing for a complete chunk

        # Iterate either till chunk size is reached or no intervals available
        while still_miss > 0:
            if not self._intervals:
                break

            (lower, upper) = self._intervals[0]
            still_miss -= upper - lower

            # Get entire interval if it fits, else only get a part of it
            if still_miss >= 0:
                out_interv.append(self._intervals.pop(0))
            else:
                split_val = upper + still_miss
                out_interv.append([lower, split_val])
                self._intervals[0] = (split_val, upper)

        # Fetch and concat pixel intervals
        [store, pixel], keys = self._uris.hdf5_uris(), self._columns
        chunks = [fetch_chunk(store, pixel, l, u, keys) for l, u in out_interv]
        chunk = pd.concat(chunks)

        # Annotate if required
        if self._annotated:
            chunk = cooler.annotate(chunk, self._bins, replace=False)

        return chunk


class RawTable:
    """Handler for a raw pixel table (copy of the original pixel table).

    Object to handle the raw pixel table, meaning the original pixel table
    simply copied to the table root location. It implements a creator method
    for instances of the `ChunksIterator` class, inherited by `HiconaTable`.
    """

    def __init__(
        self,
        uris: Uris,
        scheduler: ProcessingFlow,
        bin_size: int,
        chunk_size: int,
    ):

        self._uris = uris
        self._process_info = scheduler
        self._bin_size = bin_size
        self._chunk_size = chunk_size

    @property
    def bin_size(self) -> int:
        """Resolution of the original cooler (size of the bins in bp)."""
        return self._bin_size

    @property
    def chunk_size(self) -> int:
        """Size of the chunks to retrieve during iteration."""
        return self._chunk_size

    @property
    def process_info(self) -> ProcessingFlow:
        """ProcessingFlow object containing all processing information."""
        return self._process_info

    @property
    def uris(self) -> Uris:
        """Uris object containing all table uris."""
        return self._uris

    def _get_iterator(
        self,
        intervals: list[tuple[int, int]],
        columns: list[str] | None,
        annotated: bool,
    ) -> ChunksIterator:
        """Return chunks iterator with specified intervals and columns."""

        chunks = ChunksIterator(
            uris=self._uris,
            chunk_size=self._chunk_size,
            intervals=intervals,
            columns=columns,
            annotated=annotated,
        )
        return chunks

    def chunks(
        self,
        columns: list[str] | None = None,
        annotated: bool = False,
    ) -> ChunksIterator:
        """Returns an iterator of table chunks (as pandas DataFrames).

        Parameters
        ----------
        columns: Iterable[str], optional
            If provided, only fetch the specified columns. Default is None.
        annotated: bool, optional
            Whether to annotate with the bin information. Default is False.

        Returns
        -------
        An iterator of table chunks (as pandas DataFrames).
        """

        intervals: list[tuple[int, int]] = [(0, self.get_table_size())]
        return self._get_iterator(intervals, columns, annotated)

    def get_table_size(self) -> int:
        """Fetch the total number of pixels in the table.

        Returns
        -------
        Total number of pixels in the table as an integer.
        """

        return get_table_size(*self._uris.hdf5_uris())


class HiconaTable(RawTable):
    """Placeholder"""

    def dataframe(
        self,
        columns: Iterable[str] | None = None,
        annotated: bool = False,
    ) -> pd.DataFrame:
        """Return all table chunks in a single pandas DataFrame.

        Parameters
        ----------
        columns: Iterable[str], optional
            If provided, only fetch the specified columns. Default is None.

        Returns
        -------
        A pandas DataFrame with all table pixels.
        """

        return pd.concat(self.chunks(columns)).reset_index(drop=True)


class ToFix:
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
        modality: str = "min",
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

    def get_chunks(
        self,
        alpha: str | float = None,
    ) -> ChunksIterator:
        """Returns an iterator of table chunks (as pandas DataFrames)."""
        # TODO: also make columns selectable

        if alpha == "optimal":
            if not self._alpha_optimal:
                raise ValueError("W: ignored, compute optimal alpha first.")
            alpha = self._alpha_optimal

        alpha_query = f"{self._alpha_mod} <= {alpha}" if alpha else None
        return super().get_chunks(alpha_query)

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
        # TODO: also make columns selectable

        if alpha == "optimal":
            if not self._alpha_optimal:
                raise ValueError("W: ignored, compute optimal alpha first.")
            alpha = self._alpha_optimal

        alpha_query = f"{self._alpha_mod} <= {alpha}" if alpha else None
        return super().get_dataframe(alpha_query)

    def get_alpha_distr(self) -> pd.Series:
        """Placeholder"""

        distr = pd.Series()
        for chunk in self.get_chunks():
            vals = chunk.groupby(self._alpha_mod)["count"].count()
            distr = distr.combine(vals, lambda x, y: x + y, fill_value=0)

        return distr

    def annotation_dynamics(self, annot: str) -> pd.DataFrame:
        """Placeholder"""

        def process_table(table, lower, upper):
            """Placeholder"""

            ann1, ann2, alpha_col = table.columns  # Assumed for convenience

            filt_table = table.loc[table[alpha_col] <= upper]
            filt_table = filt_table.query(f"{alpha_col} > {lower}")

            out = filt_table.groupby([ann1, ann2]).count()
            out.reset_index(inplace=True)
            out["alpha"] = upper

            return out

        def compute_quantiles(table):
            """Placeholder"""

            curve = pd.Series()
            total = 0
            for chunk in table.get_chunks():
                total += len(chunk)
                vals = chunk.groupby("alpha_min")["count"].count()
                curve = curve.combine(vals, lambda x, y: x + y, fill_value=0)

            num_pix = curve.sum()
            cumulative = curve.cumsum()
            quantiles = [0.1 * i * num_pix for i in range(1, 11)]
            thrs = [cumulative[cumulative >= q].index[0] for q in quantiles]
            thrs = [0] + thrs  # Added after since first index is not 0

            return thrs

        alphas = compute_quantiles(self)
        interv = [(alphas[i], alphas[i + 1]) for i in range(len(alphas) - 1)]

        parent_store = self._table_uri.split("hicona_tables")[0].strip("/")
        parent_location = self._store_uri
        if parent_store:  # Non empty -> multires
            parent_location += f"::{parent_store}"

        parent_cooler = cooler.Cooler(parent_location)
        bins = parent_cooler.bins()[:]
        # TODO: slightly memory demanding, maybe just fetch subset of bins

        ann1, ann2 = f"{annot}1", f"{annot}2"
        ann_df = cooler.annotate(self.get_dataframe(), bins)
        ann_df = ann_df[[ann1, ann2, self._alpha_mod]]

        # Sort annotations alphabetically to make a triagular matrix later
        ann_df[ann1], ann_df[ann2] = np.where(
            ann_df[ann1] < ann_df[ann2],
            (ann_df[ann1], ann_df[ann2]),
            (ann_df[ann2], ann_df[ann1]),
        )

        out_df = pd.concat([process_table(ann_df, *i) for i in interv])

        out_df.reset_index(drop=True, inplace=True)
        out_df.rename(columns={self._alpha_mod: "num_pixels"}, inplace=True)

        return out_df


# THRESHOLD VERSION OF ANNOTATION DYNAMICS
# def annotation_dynamics(self, annot: str, step: float = 0.05) -> pd.DataFrame:
#     """Placeholder"""

#     def process_table(table, lower, upper):
#         """Placeholder"""

#         ann1, ann2, alpha_col = table.columns  # Assumed for convenience

#         filt_table = table.loc[table[alpha_col] <= upper]
#         filt_table = filt_table.query(f"{alpha_col} > {lower}")

#         out = filt_table.groupby([ann1, ann2]).count()
#         out.reset_index(inplace=True)
#         out["alpha"] = upper

#         return out

#     alphas = [round_half_up(n, 2) for n in np.arange(0, 1, step)]
#     interv = [(alphas[i], alphas[i + 1]) for i in range(len(alphas) - 1)]

#     parent_store = self._table_uri.split("hicona_tables")[0].strip("/")
#     parent_location = self._store_uri
#     if parent_store:  # Non empty -> multires
#         parent_location += f"::{parent_store}"

#     print(parent_location)
#     parent_cooler = cooler.Cooler(parent_location)
#     bins = parent_cooler.bins()[:]

#     ann1, ann2 = f"{annot}1", f"{annot}2"
#     ann_df = cooler.annotate(self.get_dataframe(), bins)
#     ann_df = ann_df[[ann1, ann2, self._alpha_mod]]

#     # Sort annotations alphabetically to make a triagular matrix later
#     ann_df[ann1], ann_df[ann2] = np.where(
#         ann_df[ann1] < ann_df[ann2],
#         (ann_df[ann1], ann_df[ann2]),
#         (ann_df[ann2], ann_df[ann1]),
#     )

#     out_df = pd.concat([process_table(ann_df, *i) for i in interv])

#     out_df.reset_index(drop=True, inplace=True)
#     out_df.rename(columns={self._alpha_mod: "num_pixels"}, inplace=True)

#     return out_df

# CUMULATIVE VERSION OF ANNOTATION DYNAMICS
# def annotation_dynamics(
#     self,
#     annot: str,
#     alphas: float | list[float],
# ) -> pd.DataFrame:
#     """Placeholder"""

#     def process_table(table, alpha):
#         """Placeholder"""

#         ann1, ann2, alpha_col = table.columns  # Assumed for convenience
#         table.query(f"{alpha_col} <= {alpha}", inplace=True)
#         table_size = len(table)

#         out = table.groupby([ann1, ann2]).count()  # / table_size
#         print(alpha)
#         print(out)
#         out = out / table_size
#         out.reset_index(inplace=True)
#         out["alpha"] = alpha

#         return out

#     alphas = [alphas] if isinstance(alphas, float) else alphas
#     alphas.sort(reverse=True)

#     parent_store = self._table_uri.split("hicona_tables")[0].strip("/")
#     parent_location = self._store_uri
#     if parent_store:  # Non empty -> multires
#         parent_location += f"::{parent_store}"

#     parent_cooler = cooler.Cooler(parent_location)
#     bins = parent_cooler.bins()[:]

#     ann1, ann2 = f"{annot}1", f"{annot}2"
#     ann_df = cooler.annotate(self.get_dataframe(), bins)
#     ann_df = ann_df[[ann1, ann2, self._alpha_mod]]

#     # Sort annotations alphabetically to make a triagular matrix later
#     ann_df[ann1], ann_df[ann2] = np.where(
#         ann_df[ann1] < ann_df[ann2],
#         (ann_df[ann1], ann_df[ann2]),
#         (ann_df[ann2], ann_df[ann1]),
#     )

#     out_df = pd.concat([process_table(ann_df, a) for a in alphas])
#     out_df.reset_index(drop=True, inplace=True)
#     out_df.rename(columns={self._alpha_mod: "fraction"}, inplace=True)

#     return out_df


class _TablesIterator:
    """Iterator of HiconaTable objects."""

    # TODO: Fix to because of regions and such

    def __init__(self, store, root, uris):
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

        return HiconaTable(self._store, curr_uri)
