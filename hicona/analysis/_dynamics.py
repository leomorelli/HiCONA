"""Placeholder"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy as sp

from hicona._dtypes import AlphaModType, OptionalAxes
from hicona._ops import chunked, dataf
from hicona.analysis import _plotting

if TYPE_CHECKING:
    from hicona._table import HiconaTable


__all__ = ["AnnotDynamics"]


class AnnotDynamics:
    """Class for the handling of pixel table annotation dynamics.

    Class used to compute, return and plot table annotation dynamics, that
    is, the changes in frequency of pixel annotation pairs as a function of
    the alpha value.

    At initialization, the object creates a 1 point spaced grid of alpha
    thresholds (every 1 quantile or every 1 percentage point).

    Parameters
    ----------
    table : HiconaTable
        The table for which the annotation dynamics are computed.
    annot_name : str
        The name of the annotation column to consider.
    as_quantiles : bool, optional
        Whether the intervals are expressed in quantile points, rather than
        in absolute percentage points. Default is True.
    alpha_mod : "alpha_min" or "alpha_max", optional
        The alpha mode to use for filtering. Default is "alpha_min".
    """

    def __init__(
        self,
        table: "HiconaTable",
        annot_name: str,
        as_quantiles: bool = True,
        alpha_mod: AlphaModType = "alpha_min",
    ):

        self._hic_table = table
        self._anno_name = annot_name
        self._alpha_mod = alpha_mod
        self._as_quants = as_quantiles

        # Automatically generate a grid with 0.01 wide intervals
        self._alpha_distr = self._hic_table.get_alpha_distr(alpha_mod)
        self._break_pts = self._compute_breakpoints(0.01, self._as_quants)
        self._abs_dynam = self._compute_dynamics()

    def _compute_breakpoints(self, size: float, as_quants: bool) -> list[float]:
        """Get the break points for the intervals."""

        # Define the discrete alpha break points
        num_pt: int = int(1 / size)
        points: list[float] = [size * i for i in range(1, num_pt + 1)]

        # Convert break points to quantiles if needed
        if as_quants:
            chunks = self._hic_table.chunks()
            quants = chunked.chunked_quants(chunks, self._alpha_mod, points)
            points = quants[self._alpha_mod].tolist()

        return points

    def _compute_dynamics(self) -> pd.DataFrame:
        """Compute absolute frequencies of annotation pairs per interval."""

        # Set up variables
        anno_cols = [f"{self._anno_name}1", f"{self._anno_name}2"]

        # NOTE: dedup is needed since quantiles can yield duplicate points
        points = list(np.unique(self._break_pts))
        points.reverse()

        # Retrieve annotation dynamics in chunks
        dynam_parts = []
        chunk_cols = anno_cols + [self._alpha_mod]
        for chunk in self._hic_table.chunks(annotated=True, columns=chunk_cols):

            # Have annotations alpahebetically sorted to avoid duplicates
            dataf.swap_columns(chunk, *anno_cols)

            # Compute cumulative absolute frequencies per interval per chunk
            for pt in points:
                chunk.query(f"{self._alpha_mod} <= {pt}", inplace=True)
                chunk_parts = chunk.groupby(anno_cols, as_index=False).count()
                chunk_parts["upper"] = pt
                dynam_parts.append(chunk_parts)

        # Create full dynamics table
        dynam = pd.concat(dynam_parts, ignore_index=True)
        dynam = dynam.groupby(anno_cols + ["upper"], as_index=False).sum()

        # Convert to table with annotations as rows, upper bounds as columns
        dynam = dynam.pivot_table(
            index=anno_cols,
            columns="upper",
            values=self._alpha_mod,
            fill_value=0,
        )

        # Convert to matrix to fix missing columns and such
        matrix = np.zeros([len(dynam), len(self._break_pts)], dtype=int)
        for ind, val in enumerate(self._break_pts):
            matrix[:, ind] = dynam.get(val, 0)

        # Using ordering names since duplicate columns break many ops
        col_names = [i + 1 for i in range(len(self._break_pts))]
        return pd.DataFrame(matrix, index=dynam.index, columns=col_names)

    def get_dynamics(
        self,
        cumulative: bool = False,
        intervals: list[int] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return annotation dynamics as a pd.DataFrame.

        For each annotation, compute the log odds ratio and p-value for each
        alpha interval. P-values are computed using Fisher's exact test and
        corrected for multiple testing using the Benjamini-Hochberg method.

        If set to cumulative, the column considers all alpha values up to
        the upper bound (included), otherwise it considers only alphas between
        the column upper bound (included) and the previous column upper bound
        (excluded).

        If intervals are provided, the dynamics are computed on those
        intervals rather than on the 1 unit spaced grid by default.
        Interval boundaries should be in the range 0 < i <= 100, since they
        are unit points of the underlying coarse grain grid. As an example,
        the interval [5, 10] would indicate alpha values between 0.05 and 0.1
        if working with percentiles, or between the 5th and 10th quantiles if
        working with quantiles.

        NOTE: currently pixels with alpha value equal to zero are lost since
        the lower bound is excluded. This should be changed in the future
        even though the number of pixels with alpha equal to zero is usually
        very low.

        Parameters
        ----------
        cumulative : bool, optional
            Whether to compute the cumulative dynamics. Default is False.
        intervals : list[int] or None, optional
            If provided, the upper bounds of the intervals to consider.
            Default is None.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            A tuple with the log odds ratios and p-values matrices.
        """

        dynam: pd.DataFrame = self._abs_dynam.copy()

        # Restrict the table to the intervals of interest if provided
        if intervals is not None:

            if not all(0 < i <= 100 for i in intervals):
                raise ValueError("Bounds must be in 0 < i <= 100")
            if len(set(intervals)) != len(intervals):
                raise ValueError("Upper bounds must be unique")

            dynam = dynam[sorted(intervals)]

        # Convert to non cumulative if needed
        if not cumulative:
            first_col = dynam.columns[0]
            dynam = dynam.diff(axis=1)
            dynam[first_col] = self._abs_dynam[first_col]

        # Compute log odds ratios and p-values matrices
        ref_col = self._abs_dynam[self._abs_dynam.columns[-1]]

        results = [dataf.serial_odds_ratios(dynam[c], ref_col) for c in dynam.columns]
        axes = {"columns": dynam.columns, "index": dynam.index}
        log_odds = pd.DataFrame(np.array([r[0] for r in results]).T, **axes)
        p_values = pd.DataFrame(np.array([r[1] for r in results]).T, **axes)

        # Apply FDR correction to p-values
        for _, row in p_values.iterrows():
            valid = (row >= 0) * (row <= 1)
            adj_p_values = sp.stats.false_discovery_control(row[valid])
            row[valid] = adj_p_values

        # Convert back to actual col names with duplicates
        log_odds.columns = intervals or self._break_pts  # type: ignore
        p_values.columns = intervals or self._break_pts  # type: ignore

        return log_odds, p_values

    def plot_full(
        self,
        cumulative: bool = True,
        intervals: list[int] | None = None,
        sort_rows: bool = True,
        img_path: str | None = None,
        show: bool = False,
    ) -> OptionalAxes:
        """Plot the full annotation dynamics.

        Plot the full annotation dynamics table (or a subset if specified)
        as a heatmap with all the possible annotation pairs as rows and the
        thresholds as columns. On top of the heatmap, represent the alpha
        distribution and the thresholds of the heatmap cells (dashed lines).

        Parameters
        ----------
        cumulative : bool, optional
            Whether to compute the cumulative dynamics. Default is True.
        intervals : list[int] or None, optional
            If provided, the upper bounds of the intervals to consider.
            See `get_dynamics` for more information.
            Default is None.
        sort_rows : bool, optional
            Whether to sort the rows of the heatmap by linkage. This is to try
            to group similar annotations together, though it might hamper the
            comparison of multiple tables. Default is True.
        img_path : str or None, optional
            If provided, save the plot to the specified path. Default is None.
        show : bool, optional
            If True, display the plot in a :py:mod:`matplotlib` window.
            Default is False.

        Returns
        -------
        matplotlib.axes.Axes or None
            If ``show`` if `False` and ``img_path`` is `None`, return plot
            axes. Otherwise, return None.
        """

        odds, _ = self.get_dynamics(cumulative, intervals)
        _plotting.plot_dynamics_full(
            odds,
            self._alpha_distr,
            sort_rows=sort_rows,
            inf_to_nan=True,
            img_path=img_path,
            show=show,
        )

    def plot_interval(
        self,
        lower_bound: int,
        upper_bound: int,
        img_path: str | None = None,
        show: bool = False,
    ) -> OptionalAxes:
        """Plot the annotation dynamics for a restricted p-value interval.

        Given a single alpha value interval (lower bound excluded, upper bound
        included), create a lower-triangular heatmap for the annotation
        enrichment in that interval.

        Parameters
        ----------
        lower_bound : int
            The lower bound of the alpha interval to plot. See `get_dynamics`
            for more information.
        upper_bound : int
            The upper bound of the alpha interval to plot. See `get_dynamics`
            for more information.
        img_path : str or None, optional
            If provided, path to save the plot to. (default is None)
        show : bool, optional
            If True, display the plot in a :py:mod:`matplotlib` window.
            (default is False)

        Returns
        -------
        matplotlib.axes.Axes or None
            If ``show`` if `False` and ``img_path`` is `None`, return plot
            axes. Otherwise, return None.
        """

        interv = [lower_bound, upper_bound]
        odds, pvals = self.get_dynamics(intervals=interv, cumulative=False)

        _plotting.plot_dynamics_interval(
            odds[upper_bound],
            pvals[upper_bound],
            img_path,
            show,
        )
