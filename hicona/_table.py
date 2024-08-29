"""Handle object for processed tables.

Object used to retrieve and analyze processed pixel tables.

"""

import collections

import polars as pl

from hicona._core import base_table, uris
from hicona.analysis import ThresholdGrid, AnnotDynamics


class HiconaTable(base_table.Table):
    """Class for preprocessed pixel table handling.

    Class implementing everything needed to handle preprocessed pixel tables,
    such as pixel fetching, subsetting, boolean operations and analysis.

    In general, instances of this class should be created via the
    ``HiconaCooler.fetch_table`` method rather than directly, for this reason
    the class constructor is not publicly documented.

    See Also
    --------
    HiconaCooler.fetch_table : Fetch a table from a cooler file.
    """

    def __init__(
        self,
        uris_path: uris.Uris,
        intervals: base_table.TableIndex | None = None,
    ):
        """Initialize the HiconaTable object.

        Constructor parameters documented in here to hide from the public API.

        Parameters
        ----------
        uris_path : Uris
            Uris object specifying the location of the table in the cooler file.
        intervals : TableIntervals, optional
            TableIntervals object specifying the parts of the table to consider.
            If not provided, the full table is considered. Default is None.
        """

        super().__init__(uris_path, intervals=intervals)

    @property
    def region(self) -> str:
        """Return the region of the table or the subset.

        The region is expressed as a string with a custom notation:

        - ``full_table``: if the table was directly fetched from the cooler.
        - bed-like intervals followed by ``(+)`` or ``(-)``: this means that both
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
        >>> table = handle.fetch_table("hicona")
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
        return str(self._index)

    def subset(self, region: str, both: bool = True) -> "HiconaTable":
        """Return a subset of the table based on pixel genomic regions.

        Return a new ``HiconaTable`` object with only the pixels that have
        both or at least one of their bins in the specified genomic region.
        This is useful to restrict both visualization and analysis.

        .. note::
            Currently, subsetting of a subset is not supported. This behavior
            may change in future versions. Currently, multiple subsetting can
            be achieved by using a combination of boolean operations.

        Parameters
        ----------
        region : str
            Genomic region to keep. Allowed string formats are:

            - chromosome: ``chr1``
            - bed-like interval: ``chr1 10000 200000``
            - position-like interval: ``chr1:10000-200000``

        both : bool, optional
            Whether to keep a pixel only if both of its bins are in the
            specified region. If 'False', keep a pixel if at least one of its
            bins is in the region. Default is 'True'.

        Returns
        -------
        HiconaTable
            A new HiconaTable object with the subset of the original table.

        Examples
        --------
        Subset a table based on a genomic region (both bins in interval):

        >>> handle = HiconaCooler("path/to/cool_file.cool")
        >>> table = handle.fetch_table("hicona")
        >>> print(table.size)
        449642
        >>> both = table.subset("chr1 10000 5000000")
        >>> print(both.size)
        27728

        Subset a table based on a genomic region (at least one bin in interval):

        >>> either = table.subset("chr1 10000 5000000", both=False)
        >>> print(either.size)
        428400
        """

        new_intervals = self._index.subset(region, both)
        return HiconaTable(self.uris, new_intervals)

    def _same_source_check(self, other: "HiconaTable") -> None:
        """Return whether the two tables have the same source cooler."""

        same_uris = self.uris.cooler_uri() == other.uris.cooler_uri()
        same_size = self.chunk_size == other.chunk_size

        if not same_uris or not same_size:
            raise ValueError("Tables must have the same source cooler and chunk size.")

    def __or__(self, other: "HiconaTable") -> "HiconaTable":

        self._same_source_check(other)
        new_intervals = self._index | other._index
        return HiconaTable(self.uris, new_intervals)

    def __and__(self, other: "HiconaTable") -> "HiconaTable":

        self._same_source_check(other)
        new_intervals = self._index & other._index
        return HiconaTable(self.uris, new_intervals)

    def get_threshold_grid(
        self,
        decimals: int = 3,
        verbose: bool = True,
    ) -> "ThresholdGrid":
        """Return a grid of stats for the table filtered at different score thresholds.

        Compute the number and fraction of nodes and edges remaining in the
        table when filtered at different score thresholds. Return this information
        as an ``ThresholdGrid`` object, which can also be used to plot the results
        and retrieve the optimal score threshold for filtering the table.

        For the threshold selection procedure see the ``ThresholdGrid`` class.

        Parameters
        ----------
        decimals : int, optional
            Maximum number of decimal positions for the thresholds. Default is 3.
        verbose : bool, optional
            Whether to log the progress of the computation. Default is 'True'.

        Returns
        -------
        ThresholdGrid
            An ThresholdGrid object with the statistics for each score threshold.

        See Also
        --------
        analysis.ThresholdGrid : Class for the score threshold grid computation and selection.

        Examples
        --------
        Compute the score threshold grid for a table and fetch the optimal score threshold:

        >>> handle = HiconaCooler("path/to/cool_file.cool")
        >>> table = handle.fetch_table("hicona")
        >>> score_grid = table.get_threshold_grid()
        >>> score_grid.optimal
        0.123
        """

        return ThresholdGrid(self, decimals, verbose)

    def get_annot_dynamics(
        self,
        annot: str,
        as_quantiles: bool = True,
    ):
        """Return annotation dynamics as a function of score filtering.

        Given a multimodal annotation, the dynamics of the annotation are
        the enrichments of each modality of the annotation as the table is
        filtered using progressively more stringent score thresholds.

        Given that a table is composed of pixels, meaning two bins with
        individual values for the annotation, the dynamics are computed for
        each unordered pair of modalities of the annotation, rather than for
        each modality individually.

        The function returns an instance of the ``AnnotDynamics`` class, which
        can be used to compute, return and plot the annotation dynamics. See
        object documentation for more information on parameters and outputs.

        Parameters
        ----------
        annot : str
            The name of the multimodal annotation column to consider.
        as_quantiles : bool, optional
            Whether the intervals are expressed in quantile points, rather than
            in absolute percentage points. Default is 'True'.

        Returns
        -------
        AnnotDynamics
            An AnnotDynamics object for the specified annotation and table.

        See Also
        --------
        analysis.AnnotDynamics : Class for the annotation dynamics computation.

        Examples
        --------
        Obtain an ``AnnotDynamics`` object and compute annotation dynamics for a table:

        >>> handle = HiconaCooler("path/to/cool_file.cool")
        >>> table = handle.fetch_table("hicona")
        >>> dynamics = table.get_dynamics("multimodal_annot")
        >>> log_odds, pvals = dynamics.get_dynamics()
        >>> log_odds.head()
        # TODO: remake this example
        """

        return AnnotDynamics(self, annot, as_quantiles)

    def get_distribution(self, colname: str) -> pl.DataFrame:
        """Return the distribution for the column as a DataFrame.

        Obtain the value distribution for the specified column in the table.
        The distribution is provided as a DataFrame with the columns ``value``
        and ``count`` representing the unique values and their number
        of occurrences, respectively.

        Parameters
        ----------
        colname : str
            The column for which to obtain the value distribution.

        Returns
        -------
        polars.DataFrame
            A dataframe with the distribution.

        Examples
        --------
        Compute the score distribution for a table:

        >>> handle = HiconaCooler("path/to/cool_file.cool")
        >>> table = handle.fetch_table("hicona")
        >>> score_distr = table.get_distribution("score")
        >>> score_distr.head(3)
            value  count
        0   0.000    675
        1   0.001   1245
        2   0.002   1456
        """

        # TODO: Maybe add check that it is a numeric column (e.i. no bin_id)

        counter = collections.Counter()
        for chunk in self.chunks():
            counter.update(chunk[colname].to_list())

        return pl.DataFrame({"value": counter.keys(), "count": counter.values()}).sort(
            "value"
        )
