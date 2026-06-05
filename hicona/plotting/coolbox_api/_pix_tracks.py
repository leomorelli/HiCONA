"""Custom CoolBox tracks for the visualization of pixel table features."""

from typing import TYPE_CHECKING, Literal

import coolbox.api as cp
import numpy as np

from ._compatibility import HicMatBase
from ._track_defaults import get_updated_defaults

if TYPE_CHECKING:
    from hicona import PixelTable

__all__ = ("PixelTrack", "PixelCounts", "PixelProbs", "PixelClusters")


class PixelTrack(HicMatBase):
    """Multi-purpose track for data coming from a PixelTable.

    Subclass of :class:`coolbox.api.HicMatBase`, designed to fetch data from a
    :class:`PixelTable` and display it as a matrix of pixels. For the most common
    use-cases, specialized classes are provided; instances from those classes are the
    same as instances from this class initialized with some default parameters.

    Parameters
    ----------
    pixels_table : PixelTable
        Pixel table containing the data to be displayed.
    value_col : str
        Name of the column, in the pixel table, containing the values to be displayed.
    **kwargs
        Additional parameters to be passed to the HicMatBase constructor.

    """

    def __init__(
        self,
        pixel_table: "PixelTable",
        *,
        value_col: str,
        **kwargs,
    ):

        super().__init__(**get_updated_defaults(PixelTrack.__name__, kwargs))
        self._table: PixelTable = pixel_table
        self._value_col: str = value_col

    def fetch_data(
        self,
        gr: str | cp.GenomeRange,
        gr2: str | cp.GenomeRange | None = None,
        **kwargs,
    ) -> "np.ndarray":
        """Fetch the pixels corresponding to the genomic region in a dense matrix.

        Added for compatibility with CoolBox API. When not plotting, use
        directly :meth:`PixelTable.get_matrix` instead.

        Parameters
        ----------
        gr : str or GenomeRange
            Genomic region of interest.
        gr2 : str or GenomeRange or None
            Useless, currently present only for signature compatibility with CoolBox API.
        **kwargs
            Useless, currently present only for signature compatibility with CoolBox API.

        Returns
        -------
        np.ndarray
            Dense matrix of pixel values.

        """

        if gr2 or kwargs:
            raise NotImplementedError("Currently, `gr` is the only supported parameter")

        if isinstance(gr, cp.GenomeRange):
            gr = f"{gr.chrom}:{gr.start}-{gr.end}"
        return self._table.get_matrix(gr, value_col=self._value_col)


class PixelCounts(PixelTrack):
    """Track displaying count values of the pixels in the PixelTable.

    Parameters
    ----------
    pixels_table : PixelTable
        Pixel table containing the data to be displayed.
    value_col : str, optional
        Name of the column, in the pixel table, containing the values to be displayed.
        By default, "count".
    transform : one of {"raw", "log2", "log10"}
        Which transformation to apply to the counts before plotting.
    **kwargs
        Additional parameters to be passed to the HicMatBase constructor. These will
        override any declaration of the same parameter from the default values.

    """

    def __init__(
        self,
        pixel_table: "PixelTable",
        *,
        value_col: str = "count",
        transform: Literal["raw", "log2", "log10"] = "raw",
        **kwargs,
    ):
        kwargs["transform"] = False if transform == "raw" else transform
        super().__init__(
            pixel_table,
            value_col=value_col,
            **get_updated_defaults(PixelCounts.__name__, kwargs),
        )


class PixelProbs(PixelTrack):
    """Track displaying edge probabilites assigned to each pixel in a PixelTable.

    By default, the column `probs`, if present, contains the values of the edge
    probabilites assigned to each pixel by the network reconstruction algorithm.

    Parameters
    ----------
    pixels_table : PixelTable
        Pixel table containing the data to be displayed.
    value_col : str, optional
        Name of the column, in the pixel table, containing the values to be displayed.
        By default, "probs".
    **kwargs
        Additional parameters to be passed to the HicMatBase constructor. These will
        override any declaration of the same parameter from the default values.

    """

    def __init__(
        self,
        pixel_table: "PixelTable",
        *,
        value_col: str = "probs",
        **kwargs,
    ):
        super().__init__(
            pixel_table,
            value_col=value_col,
            **get_updated_defaults(PixelProbs.__name__, kwargs),
        )


class PixelClusters(PixelTrack):
    """Track displaying a level of the hierarchical clustering for a PixelTable.

    Shows one of the levels of the hierarchical structure found via network
    reconstruction. Will not work if network reconstruction was not performed.

    Parameters
    ----------
    pixels_table : PixelTable
        Pixel table containing the data to be displayed.
    level: int
        Level of the hierarchical structure to display.
    col_name_pattern: str, optional
        String which, when formatted with the value provided as the `level` parameter,
        will coincide with the name of the column containing the clustering information
        for the specified level. By default "level_({})".
    **kwargs
        Additional parameters to be passed to the HicMatBase constructor. These will
        override any declaration of the same parameter from the default values.

    """

    def __init__(
        self,
        pixel_table: "PixelTable",
        level: int,
        *,
        col_name_pattern: str = "level_({})",
        **kwargs,
    ):

        TITLE_PATTERN: str = "Cluster\nlevel {}"

        # NOTE: Kwargs are updated with title outside of init to prevent downstream
        # issues such as "two values were provided for parameter title" etc.
        kwargs = kwargs.copy()
        kwargs.update({"title": TITLE_PATTERN.format(level)})

        super().__init__(
            pixel_table,
            value_col=col_name_pattern.format(level),
            **get_updated_defaults(PixelClusters.__name__, kwargs),
        )

        # NOTE: Level check is performed after super().__init__() else parent Track will
        # raise an exception due to __del__ trying to remove non-set property.
        level_col: str = col_name_pattern.format(level)
        if level_col not in pixel_table.bins.get_dataframe().columns:
            raise KeyError(f"Clustering level `{level}` does not exist.")
