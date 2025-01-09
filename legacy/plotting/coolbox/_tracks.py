"""Custom CoolBox tracks for HiCONA data visualization."""

from typing import Any, TYPE_CHECKING
import coolbox.api as cp
from coolbox.core.coverage.base import Coverage

import seaborn as sns
import matplotlib.pyplot as plt

from .._palettes import cluster_palette, prob_palette
import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes as axes
    from hicona import PixelTable
    from .._dtypes import TrackModality, TrackStyle


MODALITY_KWARGS: dict[str, dict] = {
    "counts": {},
    "probs": {
        "transform": False,
        "max_value": 1,
        "min_value": 0,
        "cmap": prob_palette,
    },
    "clusters": {
        "transform": False,
        "cmap": cluster_palette,
    },
}


class PixelsTrack(cp.HicMatBase):
    """Multi-purpose track for data coming from a PixelTable."""

    def __init__(self, pixels_table: "PixelTable", value_col: str, **kwargs):
        super().__init__(**kwargs)
        self._table: PixelTable = pixels_table
        self._value_col: str = value_col

    def fetch_data(self, gr: str | cp.GenomeRange, **kwargs) -> "np.ndarray":
        """Fetch the pixels corresponding to the genomic region in a dense matrix.

        Added for compatibility with CoolBox API. When not plotting, use
        directly :meth:`PixelTable.get_matrix` instead.

        Parameters
        ----------
        gr : str or GenomeRange
            Genomic region of interest.
        **kwargs
            Currently present only for signature compatibility with CoolBox API.

        Returns
        -------
        np.ndarray
            Dense matrix of pixel values.

        """

        if isinstance(gr, cp.GenomeRange):
            gr = f"{gr.chrom}:{gr.start}-{gr.end}"
        return self._table.get_matrix(gr, value_col=self._value_col)


class PixelsCoverage(Coverage):
    """Coverage used to overlay one pixel matrix on top of the other."""

    def __init__(self, pixels_table: "PixelTable", value_col: str, **kwargs):
        super().__init__({})
        self._table: PixelTable = pixels_table
        self._value_col: str = value_col

    def fetch_data(self, gr: str | cp.GenomeRange, **kwargs) -> "np.ndarray":
        """Fetch the pixels corresponding to the genomic region in a dense matrix.

        Added for compatibility with CoolBox API. When not plotting, use
        directly :meth:`PixelTable.get_matrix` instead.

        Parameters
        ----------
        gr : str or GenomeRange
            Genomic region of interest.
        **kwargs
            Currently present only for signature compatibility with CoolBox API.

        Returns
        -------
        np.ndarray
            Dense matrix of pixel values.

        """

        if isinstance(gr, cp.GenomeRange):
            gr = f"{gr.chrom}:{gr.start}-{gr.end}"
        return self._table.get_matrix(gr, value_col=self._value_col, mode="lower")

    def plot(self, ax: "axes.Axes", gr: str | cp.GenomeRange, **kwargs):
        """Plot the coverage on the given axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes where the coverage will be plotted.
        gr : str or GenomeRange
            Genomic region of interest.
        **kwargs
            Additional keyword arguments passed to the plotting function.

        """

        data = self.fetch_data(gr, **kwargs)
        print(data)
        ax.matshow(data, aspect="equal", alpha=0.5)
        # ax.add_artist(im)


def get_default_track(
    table: "PixelTable",
    value_col: str,
    modality: "TrackModality",
    other_kwargs: dict[str, Any] | None = None,
) -> PixelsTrack:
    """Returns a pixel track with default parameters for a certain modality."""

    kwargs = MODALITY_KWARGS.get(modality)
    if kwargs is None:
        kwargs = {}
        print(f"Modality {modality} not recognized, using default parameters.")

    if other_kwargs is not None:
        kwargs.update(other_kwargs)

    return PixelsTrack(table, value_col, **kwargs)
