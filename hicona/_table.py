"""Handle object for processed tables.

Object used to retrieve and analyze processed pixel tables.

"""

import collections

import pandas as pd

from hicona._core import base_table, uris
from hicona._dtypes import AlphaModType
from hicona.analysis import AlphaGrid, AnnotDynamics


class HiconaTable(base_table.Table):
    """Class for sparsified pixel table handling.

    Class implementing everything needed to handle sparsified pixel tables,
    such as pixel fetching, subsetting, boolean operations and analysis.

    In general, instances of this class should be created via the
    `HiconaCooler.fetch_table` method rather than directly.

    Parameters
    ----------
    uris_path : Uris
        Uris object specifying the location of the table in the cooler file.
    intervals : TableIntervals, optional
        TableIntervals object specifying the parts of the table to consider.
        If not provided, the full table is considered. Default is None.
    """

    def __init__(
        self,
        uris_path: uris.Uris,
        intervals: base_table.TableIntervals | None = None,
    ):
        super().__init__(uris_path, intervals=intervals)

    @property
    def region(self) -> str:
        """Return the region of the table or the subset.

        The region is expressed using a custom notation:

        - ``full_table``: if the table was directly fetched from the cooler.
        - bed-like intervals followed by `(+)` or `(-)`: this means that both
          or at least one of the bins of the pixel are in the specified region.
          This is the result of a subset operation.
        - bed-like intervals separated by boolean operators, if the region is
          the result of a boolean operation among multiple tables.

        Returns
        -------
        str
            The genomic region of the table or the subset.

        Examples
        --------
        Full table fetched directly from the cooler:

        >>> handle = HiconaCooler("path/to/cool_file.cool")
        >>> table = handle.fetch_table("pixels")
        >>> table.region
        'full_table'

        Subset of the table based on a genomic region:

        >>> tab_a = table.subset("chr1 10000 200000")
        >>> tab_a.region
        'chr1 10000 200000(+)'  # Both bins of the pixel are in the region

        >>> tab_b = table.subset("chr2 10000 200000", both=False)
        >>> tab_b.region
        'chr2 10000 200000(-)'  # At least one bin is in the region

        Boolean operation between two tables:

        >>> tab_c = tab_a | tab_b
        >>> tab_c.region
        'chr1 10000 200000(+) | chr2 10000 200000(-)'  # Union of the regions
        """
        return str(self._intervals)

    def subset(self, region: str, both: bool = True) -> "HiconaTable":
        """Return a subset of the table based on pixel genomic regions.

        Parameters
        ----------
        region : str
            Genomic region to keep. It must be a string in bed-like format,
            such as "chr1 10000 200000" or just "chr1".
        both : bool, optional
            Whether to keep a pixel only if both of its bins are in the
            specified region. If False, keep a pixel if at least one of its
            bins is in the region. Default is True.

        Returns
        -------
        HiconaTable
            A new HiconaTable object with the subset of the original table.
        """

        new_intervals = self._intervals.subset(region, both)
        return HiconaTable(self.uris, new_intervals)

    def _same_source_check(self, other: "HiconaTable") -> None:
        """Return whether the two tables have the same source cooler."""

        same_uris = self.uris.cooler_uri() == other.uris.cooler_uri()
        same_size = self.chunk_size == other.chunk_size

        if not same_uris or not same_size:
            raise ValueError("Tables must have the same source cooler and chunk size.")

    def __or__(self, other: "HiconaTable") -> "HiconaTable":

        self._same_source_check(other)
        new_intervals = self._intervals | other._intervals
        return HiconaTable(self.uris, new_intervals)

    def __and__(self, other: "HiconaTable") -> "HiconaTable":

        self._same_source_check(other)
        new_intervals = self._intervals & other._intervals
        return HiconaTable(self.uris, new_intervals)

    def get_alpha_grid(
        self,
        decimals: int = 3,
        verbose: bool = True,
        alpha_mod: AlphaModType = "alpha_min",
    ) -> "AlphaGrid":
        """Return a grid for the table filtered at different alpha values.

        Compute the number and fraction of nodes and edges remaining in the
        table when filtered at different alpha values. Return this information
        as an `AlphaGrid` object, which can also be used to plot the results
        and retrieve the optimal alpha value for filtering the table.

        For the threshold selection procedure see the `AlphaGrid` class.

        Parameters
        ----------
        alpha_mod : "alpha_min" or "alpha_max", optional
            The alpha mode to use for filtering. Default is "alpha_min".
        decimals : int, optional
            The number of decimal positions to consider when computing the
            grid. Default is 3.
        verbose : bool, optional
            Whether to log the progress of the computation. Default is True.

        Returns
        -------
        AlphaGrid
            An AlphaGrid object with the statistics for each alpha value.

        Examples
        --------
        Compute the alpha grid for a table and fetch the optimal alpha value:

        >>> handle = HiconaCooler("path/to/cool_file.cool")
        >>> table = handle.fetch_table("hicona")
        >>> alpha_grid = table.get_alpha_grid()
        >>> alpha_grid.optimal_alpha
        0.123
        """

        return AlphaGrid(self, alpha_mod, decimals, verbose)

    def get_annot_dynamics(
        self,
        annot: str,
        as_quantiles: bool = True,
        alpha_mod: AlphaModType = "alpha_min",
    ):
        """Return annotation dynamics as a function of alpha filtering.

        Given a multimodal annotation, the dynamics of the annotation are
        the enrichments of each modality of the annotation as the table is
        filtered using progressively more stringent alpha values.

        Given that a table is composed of pixels, meaning two bins with
        individual values for the annotation, the dynamics are computed for
        each unordered pair of modalities of the annotation, rather than for
        each modality individually.

        The function returns an instance of the `AnnotDynamics` class, which
        can be used to compute, return and plot the annotation dynamics. See
        object documentation for more information on parameters and outputs.

        Parameters
        ----------
        annot : str
            The name of the multimodal annotation column to consider.
        as_quantiles : bool, optional
            Whether the intervals are expressed in quantile points, rather than
            in absolute percentage points. Default is True.
        alpha_mod : "alpha_min" or "alpha_max", optional
            The alpha mode to use for filtering. Default is "alpha_min".

        Returns
        -------
        AnnotDynamics
            An AnnotDynamics object for the specified annotation and table.

        Examples
        --------
        Compute the annotation dynamics for a table:

        >>> handle = HiconaCooler("path/to/cool_file.cool")
        >>> table = handle.fetch_table("hicona")
        >>> dynamics = table.get_annot_dynamics("multimodal_annot")
        >>> log_odds, pvals = dynamics.compute_dynamics()
        """

        # TODO: Add check that it is a multimodal annotation

        return AnnotDynamics(self, annot, as_quantiles, alpha_mod)

    def get_alpha_distr(self, alpha_mod: AlphaModType = "alpha_min") -> pd.DataFrame:
        """Return alpha distribution in the table as a pd.DataFrame.

        The distribution is provided as a pd.DataFrame with the columns
        "value" and "count" representing the alpha values and their counts.

        Parameters
        ----------
        alpha_mod : "alpha_min" or "alpha_max", optional
            The alpha mode to use for filtering. Default is "alpha_min".

        Returns
        -------
        pd.DataFrame
            A DataFrame with the alpha distribution.

        Examples
        --------
        Compute the alpha distribution for a table:

        >>> handle = HiconaCooler("path/to/cool_file.cool")
        >>> table = handle.fetch_table("hicona")
        >>> alpha_distr = table.get_alpha_distr()
        >>> alpha_distr.head()
            value  count
        0   0.000    675
        1   0.001   1245
        2   0.002   1456

        The distribution can be plotted directly using the `plot` method:

        >>> alpha_distr.plot(x="value", y="count")
        """

        counter = collections.Counter()
        for chunk in self.chunks():
            counter.update(chunk[alpha_mod].tolist())

        distr = pd.DataFrame(counter.items(), columns=["value", "count"])
        return distr.sort_values("value")
