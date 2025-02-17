"""Custom CoolBox tracks for the visualization of bin table features."""

from typing import Any, Optional, TYPE_CHECKING

import coolbox.api as cp
from coolbox.core.track.base import Track
import polars as pl
import seaborn as sns

from ._track_defaults import get_updated_defaults
from ._track_utils import get_label_axis

if TYPE_CHECKING:
    from hicona import BinTable

__all__ = ("BinTrack", "BinClusters")


class BinTrack(Track):
    """Base class for BinTable-based tracks.

    Implements data fetching from a BinTable. Return the data corresponding to the
    provided interval for all columns in the BinTable.
    Does not implement a `plot` method, since bin columns could be of several types;
    the class must be subclassed and the method `plot` must be implemented according to
    the coolbox API in order to be used.

    Parameters
    ----------
    bin_table : BinTable
        Bin table containing the data to be displayed.

    """

    def __init__(self, bin_table: "BinTable", **kwargs) -> None:
        super().__init__(**kwargs)
        self._table: BinTable = bin_table
        self.label_ax = None

    def fetch_data(
        self,
        gr: str | cp.GenomeRange,
        **kwargs,
    ) -> Optional["pl.DataFrame"]:
        """Fetch the bins corresponding to the genomic region.

        Added for compatibility with CoolBox API. When not plotting, use
        directly :meth:`BinTable.get_dataframe` instead.

        Parameters
        ----------
        gr : str or GenomeRange
            Genomic region of interest.

        Returns
        -------
        pl.DataFrame
            DataFrame of bin values.

        """

        if isinstance(gr, cp.GenomeRange):
            gr = f"{gr.chrom}:{gr.start}-{gr.end}"
        try:
            return self._table.get_dataframe(gr)
        except AssertionError:
            return None


class BinClusters(BinTrack):
    """Track to visualize a level of bin hierarchical clustering."""

    def __init__(
        self,
        bin_table: "BinTable",
        level: int,
        *,
        col_name_pattern: str = "level_({})",
        **kwargs,
    ) -> None:

        super().__init__(
            bin_table,
            **get_updated_defaults(
                BinClusters.__name__,
                kwargs,
            ),
        )

        level_col: str = col_name_pattern.format(level)
        if level_col not in bin_table.get_dataframe().columns:
            raise KeyError(f"Clustering level `{level}` does not exist.")

        self._level = level
        self._col_name_pattern = col_name_pattern

    def plot(self, ax: Any, gr: str | cp.GenomeRange, **kwargs) -> None:
        """Plot"""

        TITLE_PATTERN: str = "level {}"

        data: pl.DataFrame | None = self.fetch_data(gr)
        if data is None:
            return None

        sns.heatmap(
            data.select(self._col_name_pattern.format(self._level)).to_numpy().T,
            ax=ax,
            cbar=False,
            cmap="tab20",
        )

        get_label_axis(ax).text(
            0.05,
            0.5,
            TITLE_PATTERN.format(self._level),
            size="large",
            horizontalalignment="left",
            verticalalignment="center",
        )
