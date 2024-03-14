"""Placeholder

Placeholder
"""

from collections.abc import Iterable
from math import dist
import json

import cooler
import numpy as np
import pandas as pd

import hicona.hicona_cooler as hicooler  # For circular import
from .uris import Uris
from .utils.dtypes import PdChunks
from .utils.hdf5_ops import fetch_chunk, get_attrs, get_table_size
from .utils.misc import build_query
from .utils.numeric import round_half_up
from .processing.processing_flow import ProcessingFlow


FULL_TABLE = "full_table"


class TableIntervals:
    """Placeholder"""

    def __init__(
        self,
        table: "Table",
        indexes: list[pd.Index] | None = None,
        repr_str: str | None = None,
    ):

        if bool(indexes) != bool(repr_str):
            raise ValueError("Both indexes and repr_str must be provided or neither.")

        self._table = table
        self._indexes = indexes or self._initial_index()
        self._repr_str = repr_str or FULL_TABLE
        self._size = 0

        self.update_size()

    def __str__(self) -> str:

        return self._repr_str

    def __or__(self, other: "TableIntervals") -> "TableIntervals":

        iterator = zip(self._indexes, other.get_indexes())
        new_intervals = [i.join(j, how="outer") for i, j in iterator]
        new_repr = f"({self._repr_str} | {other._repr_str})"

        return TableIntervals(self._table, new_intervals, new_repr)

    def __and__(self, other: "TableIntervals") -> "TableIntervals":

        iterator = zip(self._indexes, other.get_indexes())
        new_intervals = [i.join(j, how="inner") for i, j in iterator]
        new_repr = f"({self._repr_str} & {other._repr_str})"

        return TableIntervals(self._table, new_intervals, new_repr)

    def _initial_index(self) -> list[pd.Index]:
        """Return an index of the complete table split into chunks."""

        table_size = get_table_size(*self._table.uris.hdf5_uris())
        chunk_size = self._table.chunk_size

        num_full_chunks, partial_chunk_size = divmod(table_size, chunk_size)
        full_chunk_ind = pd.Index(range(0, chunk_size), dtype="int32")
        part_chunk_ind = pd.Index(range(0, partial_chunk_size), dtype="int32")

        return [full_chunk_ind] * num_full_chunks + [part_chunk_ind]

    @property
    def size(self) -> int:
        """Return the size of the complete table or the subset."""
        return self._size

    def update_size(self):
        """Update the size of the complete table or the subset."""
        self._size = sum(len(c) for c in self._indexes)

    def subset(self, region: str, both: bool = True) -> "TableIntervals":
        """Placeholder"""

        if self._repr_str != FULL_TABLE:
            raise ValueError("Cannot subset a subset. Use boolean operators instead.")

        new_repr = f"{region}({'+' if both else '-'})"

        query = build_query(region, both)
        new_indexes = [c.query(query).index for c in self._table.chunks(annotated=True)]

        return TableIntervals(self._table, new_indexes, new_repr)

    def get_indexes(self) -> Iterable[pd.Index]:
        """Return the indexes of the table or the subset."""
        for index in self._indexes:
            yield index


class Table:
    """Base class for all table types in Hicona."""

    def __init__(
        self,
        uris: Uris,
        intervals: TableIntervals | None = None,
        bin_size: int | None = None,
        chunk_size: int | None = None,
    ):

        def reconstruct_flow(uris: Uris) -> ProcessingFlow:
            """Reconstruct the ProcessingFlow object from the store."""

            # TODO: check whether the table is valid and skip if not
            tab_attrs = get_attrs(*uris.hdf5_uris())
            flow_json = json.loads(tab_attrs["process_info"])
            return ProcessingFlow.from_json(flow_json)

        def get_bin_size(uris: Uris) -> int:
            """Return the bin size of the cooler."""

            parent_cool = hicooler.HiconaCooler(uris.cooler_uri())
            return parent_cool.binsize

        def get_chunk_size(uris: Uris) -> int:
            """Return the default chunk size for the table."""

            parent_cool = hicooler.HiconaCooler(uris.cooler_uri())
            return parent_cool.chunk_size

        self._uris = uris
        self._flow = reconstruct_flow(uris)
        self._bin_size = bin_size or get_bin_size(uris)
        self._chunk_size = chunk_size or get_chunk_size(uris)
        self._intervals = intervals or TableIntervals(self)

    @property
    def bin_size(self) -> int:
        """Resolution of the original cooler (size of the bins in bp)."""
        return self._bin_size

    @property
    def chunk_size(self) -> int:
        """Size of the chunks to retrieve during iteration."""
        return self._chunk_size

    @property
    def flow(self) -> ProcessingFlow:
        """ProcessingFlow object containing all processing information."""
        return self._flow

    @property
    def uris(self) -> Uris:
        """Uris object containing all table uris."""
        return self._uris

    @property
    def size(self) -> int:
        """Total number of pixels in the table."""
        return self._intervals.size

    def chunks(
        self,
        columns: Iterable[str] | None = None,
        annotated: bool = False,
    ) -> PdChunks:
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

        def prepare_chunk(chunk, bins=None, columns=None) -> pd.DataFrame:
            """Prepare the chunk for output with annotation of filtering."""

            if bins is not None:
                chunk = cooler.annotate(chunk, bins)
            if columns:
                chunk = chunk[columns]

            return chunk

        cool = hicooler.HiconaCooler(self.uris.cooler_uri())
        bins = cool.bins()[:] if annotated else None

        chunk_parts: list[pd.DataFrame] = []

        indexes = self._intervals.get_indexes()
        for num, index in enumerate(indexes):

            if len(index) == 0:
                continue

            lower = num * self.chunk_size
            upper = (num + 1) * self.chunk_size
            chunk = fetch_chunk(*self.uris.hdf5_uris(), lower, upper)
            chunk = chunk.iloc[index]

            chunk_parts.append(chunk)

            new_chunks_size = sum(len(c) for c in chunk_parts)
            if new_chunks_size >= self.chunk_size:

                out_chunk = pd.concat(chunk_parts)

                chunk_parts = [out_chunk.iloc[self.chunk_size :]]
                out_chunk = out_chunk.iloc[: self.chunk_size]

                yield prepare_chunk(out_chunk, bins, columns)

        if len(chunk_parts) > 0:
            yield prepare_chunk(pd.concat(chunk_parts), bins, columns)

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
        annotated: bool, optional
            Whether to annotate with the bin information. Default is False.

        Returns
        -------
        A pandas DataFrame with all table pixels.
        """

        iterator = self.chunks(columns, annotated)
        return pd.concat(iterator).reset_index(drop=True)


class RawTable(Table):
    """Placeholder"""

    def reset_index(self):
        """Placeholder"""

        self._intervals = TableIntervals(self)
        # TODO: check because this might not work as expected
        # due to the way the table is resized in the hdf5 file


class HiconaTable(Table):
    """Placeholder"""

    def __init__(self, uris: Uris, intervals: TableIntervals | None = None):
        super().__init__(uris, intervals=intervals)

    # TODO: Redefine constructor to mask params

    @property
    def region(self):
        """Return the region of the table or the subset."""
        return str(self._intervals)

    def subset(self, region: str, both: bool = True) -> "HiconaTable":
        """Return a subset of the table based on the region and both strands."""

        new_intervals = self._intervals.subset(region, both)
        return HiconaTable(self.uris, new_intervals)

    def __or__(self, other: "HiconaTable") -> "HiconaTable":

        new_intervals = self._intervals | other._intervals
        return HiconaTable(self.uris, new_intervals)

    def __and__(self, other: "HiconaTable") -> "HiconaTable":

        new_intervals = self._intervals & other._intervals
        return HiconaTable(self.uris, new_intervals)


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
    ):
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
