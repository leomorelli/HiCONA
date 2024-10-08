"""Placeholder"""

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl

from hicona._dtypes import OptionalAxes
from hicona._ops import numeric
from hicona.analysis import _plotting

if TYPE_CHECKING:
    from hicona._core import PixelTable


__all__ = ["ThresholdGrid"]


class ThresholdGrid:
    """Class to analyze table filtering at different score thresholds.

    During initialization, a grid of score thresholds is tested in order
    to compute the optimal score threshold for table filtering. The optimal
    threshold value is the one minimizing retained edge fraction while
    maximizing retained node fraction.

    The computation of the grid is performed as follows:

    - define a grid of score values centered around the current optimal value
      and spaced in order to cover 1 unit of the previous decimal position.
    - for each score value, filter the table and compute the retained
      fraction of edges and nodes. An edge is retained if both nodes are
      still present in the table, while a node is retained if it is present
      in at least one edge.
    - for each score threshold compute the Euclidean distance from the point
      ``(1, 0)`` in the space with ``x: fraction of nodes``, ``y: fraction of edges``.
    - set as new optimal score the one minimizing the Euclidean distance.
    - move to the next decimal position and repeat the procedure.

    Objects of this class can also be initialized through ``HiconaTable.get_score_grid()``.

    Parameters
    ----------
    table : HiconaTable
        The table for which the score grid is computed.
    decimals : int
        The number of decimal positions to consider when computing the grid.
    verbose : bool
        Whether to log the progress of the computation.

    See Also
    --------
    hicona.HiconaTable.get_threshold_grid:
        Method to create an instance of the class starting from a table object.

    Examples
    --------
    Create an instance of the class starting from a table object:

    >>> import hicona
    >>> handle = hicona.HiconaCooler("path/to/cool_file.cool")
    >>> table = handle.fetch_table("hicona")
    >>> grid = table.get_threshold_grid()
    Optimal threshold: computing decimal 1
    Optimal threshold: computing decimal 2
    Optimal threshold: computing decimal 3
    >>> grid.optimal
    0.154

    Create an instance of the class directly:

    >>> grid = hicona.analysis.ThrehsoldGrid(table, 3, verbose=True)
    Optimal threshold: computing decimal 1
    Optimal threshold: computing decimal 2
    Optimal threshold: computing decimal 3
    >>> grid.optimal
    0.154

    """

    def __init__(
        self,
        table: "PixelTable",
        decimals: int,
        verbose: bool,
    ):
        self._table = table
        self._grid = self._compute_grid(decimals, verbose)
        self._optimal = self._get_minimal_dist(self._grid, decimals)

    def _get_stats(self, threshold: float) -> tuple[int, int]:
        """Return number of nodes and edges in a table filtered by score."""

        nodes: set = set()
        edges: int = 0

        for chunk in self._table.chunks():
            chunk = chunk.filter(pl.col("score") <= threshold)

            nodes |= set(chunk["bin1_id"]) | set(chunk["bin2_id"])
            edges += len(chunk)

        return len(nodes), edges

    def _get_score_pts(self, thresholds: list[float]) -> pd.DataFrame:
        """Return dataframe with filtering statistics for a threshold grid."""
        # NOTE: Assumed that score thresholds are in decreasing order.

        res_list = []

        # Define normalization values
        tot_nodes, tot_edges = self._get_stats(1)

        # Iterate over each threshold value
        for threshold in thresholds:
            num_nodes, num_edges = self._get_stats(threshold)
            res_list.append(
                {
                    "threshold": threshold,
                    "nodes_n": num_nodes,
                    "edges_n": num_edges,
                    "nodes_f": (frac_nodes := num_nodes / tot_nodes),
                    "edges_f": (frac_edges := num_edges / tot_edges),
                    "eu_dist": math.dist((1, 0), (frac_nodes, frac_edges)),
                }
            )

        return pd.DataFrame(res_list)

    def _compute_grid(self, decimals: int, verbose: bool) -> pd.DataFrame:
        """Return the grid containing the statistics for each score threshold."""

        # Initialize optimal score and result container
        opt_score = 0.5  # Middle of initial search space 0-1
        num_points = 21  # Must be 10x + 1 to guarantee 1 unit span
        score_vals = []

        for pos in range(decimals):
            if verbose:
                print(f"Optimal threshold: computing decimal {pos+1}")

            # Define the grid of score values to test
            step = 0.5 * 10 ** (-pos)
            grid = np.linspace(opt_score + step, opt_score - step, num_points)
            grid = [a for a in grid if 0 <= a <= 1]

            # Compute the new statistics, then update optimal score
            new_pts = self._get_score_pts(grid)
            opt_score = self._get_minimal_dist(new_pts, decimals)
            score_vals.append(new_pts)

        return pd.concat(score_vals).reset_index()

    def _get_minimal_dist(self, table: pd.DataFrame, decimals: int) -> float:
        """Return the score threshold value minimizing the Euclidean distance."""

        position = table["eu_dist"].idxmin()
        if not isinstance(position, int):
            raise ValueError("Non numeric index. This should not happen.")
        threshold_value = table["threshold"].iloc[position]

        return numeric.round_half_up(threshold_value, decimals)

    @property
    def optimal(self) -> float:
        """Return the optimal score threshold for filtering the pixel table.

        The optimal score threshold is the filtering threshold that minimizes the
        fraction of retained edges while maximizing the fraction of retained
        nodes. For process details, see class documentation.

        Returns
        -------
        float :
            The optimal score threshold.

        Examples
        --------
        Create a grid and return the optimal score threshold:

        >>> import hicona
        >>> handle = hicona.HiconaCooler("path/to/cool_file.cool")
        >>> table = handle.fetch_table("hicona")
        >>> grid = table.get_threshold_grid()
        Optimal threshold: computing decimal 1
        Optimal threshold: computing decimal 2
        Optimal threshold: computing decimal 3
        >>> grid.optimal
        0.154
        """

        return self._optimal

    def plot(
        self,
        img_path: str | None = None,
        show: bool = False,
    ) -> OptionalAxes:
        """Plot the score threshold grid.

        Plot and/or show the grid of score threshold tested when computing the
        optimal threshold score for pixel filtering. The plot has the fraction
        of retained nodes on the ``x`` axis and the fraction of retained
        edges on the ``y`` axis. The optimal threshold score is highlighted.

        Parameters
        ----------
        img_path : str or None, optional
            If provided, path to save the plot to. Default is 'None'.
        show : bool, optional
            If True, display the plot in a ``matplotlib`` window.
            Default is 'False'.

        Returns
        -------
        matplotlib.axes.Axes or None :
            If ``show`` if 'False' and ``img_path`` is 'None', return plot
            axes. Otherwise, return 'None'.

        Examples
        --------

        Generate a grid:

        >>> import hicona
        >>> handle = hicona.HiconaCooler("path/to/cool_file.cool")
        >>> table = handle.fetch_table("hicona")
        >>> grid = table.get_threshold_grid()

        Save the grid to a file:

        >>> grid.plot("threshold_grid.png")

        Get plot axes (to further customize or insert in a multi-panel plot):

        >>> ax = grid.plot()

        Display the grid in a window:

        >>> grid.plot(show=True)

        Which will display the following plot:

        .. image:: ../../_static/threshold_grid.png

        """

        _plotting.plot_threshold_grid(self._grid, img_path, show)
