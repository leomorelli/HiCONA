"""Placeholder
Placeholder
"""

import re

from cooler import Cooler, annotate
from cooler.core import delete
from cooler.util import open_hdf5
from networkx import from_pandas_edgelist, to_pandas_edgelist
from numpy import full, log2, where
from pandas import concat, DataFrame, get_dummies, Series
from pybedtools import BedTool

from .iterators import FixedSizeIterator, ChromTablesIterator
from .utils import console_log, compute_alpha_val, from_df_to_sarrays

__all__ = ["HiconaCooler"]


# Do not allow chunk size values smaller than these
_MIN_CHUNK_PIX = 1_000_000
_MIN_CHUNK_IDS = 1_000
# Template to name groups inside chrom_tables group
_GROUP_TEMPLATE = "countThr_{}_distThr_{}_stat_{}"
# Dictionary of standard regular expressions to simplify chromosome fetching
_DEFAULT_CHROM_RE = {
    "humanCanonical": "^chr([1-9]|[1][0-9]|[2][0-2]|[XY])$",
    "mouseCanonical": "^chr([1-9]|[1][0-9]|[XY])$",
}
# Table columns and respective datatypes to initialize for new table columns
_TABLE_COLS = {
    "bin1_id": "<i8",
    "bin2_id": "<i8",
    "count": "<i4",
    "dist_bin": "<i8",
    "dist_norm": "<f8",
    "spar_alpha": "<f8",
}


class HiconaCooler(Cooler):
    """An extension of Cooler objects to prepare data for network analysis.

    :py:class:`HiconaCooler` inherits from :py:class`cooler.Cooler` and
    extends it by adding new functionalities, mainly revolving around
    the creation of chromosome-level processed tables to use for network
    analyses. Tables are stored in a separate group (``chrom_tables``)
    of the :py:class:`h5py.File` and no method or property of the
    :py:class`Cooler` is overwritten, therefore a :py:class:`HiconaCooler`
    object can always be used as a :py:class`Cooler` one.

    Parameters
    ----------
    See :py:class:`cooler.Cooler` for class constructor parameters.

    Notes
    -----
    Chromosome tables are created using :py:meth:`create_tables` and can
    be accessed through :py:meth:`tables`. To print to console the list
    of available tables use :py:meth:`tables_info`.
    """

    # ////////////////////////////////////////////////////////////////////////
    # //////////////////////// BASIC OBJECT FUNCTIONS ////////////////////////
    # /////// Class constructor, setter, getters and similar functions ///////
    # ////////////////////////////////////////////////////////////////////////

    def __init__(self, *args, chunk_size: int = 1_000_000, **kwargs):
        """Extend Cooler class constructor to include processing defaults."""
        super().__init__(*args, **kwargs)
        self._chunk_size = chunk_size

    @property
    def chunk_size(self):
        """Size of fixed lenght chunks used during processing."""
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, value):
        if not (isinstance(value, int)) or value < _MIN_CHUNK_PIX:
            raise ValueError(f"chunk_size must be: int >= {_MIN_CHUNK_PIX}.")
        self._chunk_size = value

    # ////////////////////////////////////////////////////////////////////////
    # ///////////////////////// GENERATOR FUNCTIONS //////////////////////////
    # //////// Functions returning chunks of various types of tables /////////
    # ////////////////////////////////////////////////////////////////////////

    def _pix_bound_ids(self, extent: tuple):
        """Retrieve the indexes of the boundary pixels of an interval."""

        with open_hdf5(self.store, mode="r") as h5_handle:
            h5_grp = h5_handle[self.root]
            min_ind = h5_grp["indexes/bin1_offset"][extent[0]]
            max_ind = h5_grp["indexes/bin1_offset"][extent[1]]
        return (min_ind, max_ind)

    def _ids_chunk_size(self, chrom_id: str):
        """Define number of bin ids to keep per chunk.

        The number of ids is a constant computed from the proportion
        chunk_size : num_pix = ids_chunk : num_ind
        Constant number of ids does NOT lead to constant number of pixels;
        due to matrix sorting, the number of pixels per chunk decreases.
        """
        # TODO: experiment with size check but it is probably slow

        ind_bounds = self.extent(chrom_id)
        pix_bounds = self._pix_bound_ids(ind_bounds)
        num_ind = ind_bounds[1] - ind_bounds[0]
        num_pix = pix_bounds[1] - pix_bounds[0]
        return max(self._chunk_size * num_ind // num_pix, _MIN_CHUNK_IDS)

    def _full_pix_chunks(self, chrom_id: str):
        """Generator function for fixed-size pixel chunks of a chromosome."""
        # TODO: Maybe substitute entirely with chunking by bin1_id

        extent = self.extent(chrom_id)
        bound_ids = self._pix_bound_ids(extent)
        bounds = FixedSizeIterator(bound_ids, self._chunk_size)
        for lower, upper in bounds:
            yield self.pixels()[lower:upper]

    def _bin1_id_chunks(self, chrom_id: str):
        """Generator function for pixel chunks split by range of bin1_ids."""

        ind_bounds = self.extent(chrom_id)
        number_ids = self._ids_chunk_size(chrom_id)
        bin1_id_pairs = FixedSizeIterator(ind_bounds, number_ids)
        for id_pair in bin1_id_pairs:
            lower, upper = self._pix_bound_ids(id_pair)
            yield self.pixels()[lower:upper]

    def _filtered_chunks(self, chrom_id: str, indexer: Series):
        """Placeholder"""

        lo_pix = 0
        for chunk_df in self._full_pix_chunks(chrom_id):
            hi_pix = lo_pix + chunk_df.shape[0]
            filtered_df = chunk_df[indexer[lo_pix:hi_pix].values]
            lo_pix = hi_pix
            yield filtered_df

    def _table_chunks(self, table_grp: str, cols_to_get: list[str] = None):
        """Generator function for chromosome table chunks of selected size."""

        num_pixels = table_grp.attrs["size"]
        bounds = FixedSizeIterator((0, num_pixels), self._chunk_size)
        for lower, upper in bounds:
            data_dict = {c: table_grp[c][lower:upper] for c in cols_to_get}
            chunk_df = DataFrame(data_dict)
            yield chunk_df

    def _distances_chunks(self, table_grp: str):
        """Placeholder"""

        id_columns = ["bin1_id", "bin2_id"]
        for chunk_df in self._table_chunks(table_grp, id_columns):
            chunk_dist = chunk_df[id_columns[1]] - chunk_df[id_columns[0]]
            yield chunk_dist.to_frame(name="dist_bin")

    # ////////////////////////////////////////////////////////////////////////
    # /////////////////////// PRE-PROCESSING FUNCTIONS ///////////////////////
    # // Functions to pass from full-pixel table to chromosome-level tables //
    # ////////////////////////////////////////////////////////////////////////

    @console_log
    def _index_filter(self, chrom_id: str, count_thr: int, dist_thr: int):
        """Filter out pixels non conformant to some condition.

        Remove all pixels that do NOT satisfy at least one of these filters:
        - distance among the bins is below maximal genomic distance allowed
        - number of counts of the interaction is above the minimal threshold
        - the bins are distinc (non-self looping)
        - the second bin does match the provided chromosome id

        Parameters
        ----------
        pix_df : :py:class:`DataFrame`
            Dataframe of pixels to filter.
        chrom_id : str
            Id of the chromosome whose internal pixels should be kept.
        count_thr : int
            Threshold for pixel count; only keep pixels with count greater
            than this value.
        dist_thr : int
            Threshold for genomic distance; only keep pixels whose bins are
            closer to each other than this distance.

        Notes
        -----
        It is assumed that the table to filter was obtained through
        :py:meth:`Cool.pixels.fetch` and therefore that a) ``bin1_id`` field
        always matches the chromosome of interest, b) there is no ``bin2_id``
        lower than the smallest possible bin id for the specified chromosome.
        """
        # TODO: Consider directly returning it in bin_id based chunks

        max_bin_diff = -(-dist_thr // self.binsize)
        upper_bin_id = self.extent(chrom_id)[1]
        lower, upper = self._pix_bound_ids(self.extent(chrom_id))
        index_length = upper - lower

        indexer = Series(full(index_length, False), index=range(lower, upper))
        chunks_iter = self._full_pix_chunks(chrom_id)
        lo_pix = 0

        for chunk_df in chunks_iter:
            hi_pix = lo_pix + chunk_df.shape[0]
            indexer[lo_pix:hi_pix] = ~(
                (chunk_df["bin2_id"] - chunk_df["bin1_id"] >= max_bin_diff)
                | (chunk_df["count"] <= count_thr)
                | (chunk_df["bin1_id"] == chunk_df["bin2_id"])
                | (chunk_df["bin2_id"] >= upper_bin_id)
            )
            lo_pix = hi_pix

        return indexer

    @console_log
    def _deduplicate_index(self, indexer: Series, chrom_id: str):
        """Placeholder"""
        # TODO: Add note in documentation that it should not happen

        chunks_iter = self._bin1_id_chunks(chrom_id)
        tot_dups = 0
        lo_pix = 0

        for chunk_df in chunks_iter:
            hi_pix = lo_pix + chunk_df.shape[0]
            is_duplicate = chunk_df.duplicated(subset=["bin1_id", "bin2_id"])
            tot_dups += sum(is_duplicate)
            indexer[lo_pix:hi_pix] = indexer[lo_pix:hi_pix] & (~is_duplicate)
            lo_pix = hi_pix

        if tot_dups != 0:
            print(f"Warning, {tot_dups} duplicate (unfiltered) pixels found.")

    def _initialize_table(self, chrom_id: str, table_grp: str, table_size: int):
        """Placeholder"""
        # TODO: change typing of table_grp since it is not really a string

        # Save the dataframe columns as individual 1D-arrays (for storage)
        chrom_table = table_grp.create_group(chrom_id)
        chrom_table.attrs["size"] = table_size
        for col_name, col_type in _TABLE_COLS.items():
            chrom_table.create_dataset(col_name, (table_size,), dtype=col_type)

        return chrom_table

    def _save_to_table(
        self,
        chrom_table: str,
        chunks_generator,
        cols_to_save: list[str] = None,
    ):
        """Placeholder"""
        # TODO: change typing of table_grp since it is not really a string
        # TODO: Find a way to add typing to generator
        lo_pix = 0
        for chunk_df in chunks_generator:
            hi_pix = lo_pix + chunk_df.shape[0]

            # If no columns are specified, default to all of them
            if not cols_to_save:
                cols_to_save = chunk_df.columns.tolist()

            # Save df columns to the appropriate datasets
            for col_name in cols_to_save:
                table_col = chrom_table[col_name]
                table_col[lo_pix:hi_pix] = chunk_df[col_name]

            lo_pix = hi_pix

    def _add_bin_dist(self, chrom_table):
        """Placeholder"""

        dist_chunks = self._distances_chunks(chrom_table)
        self._save_to_table(chrom_table, dist_chunks)

    def _add_norm_dist(self, chrom_table):
        """Placeholder"""

        cols_of_int = ["dist_bin", "count"]
        # Create the chunks
        # TODO: "count" column is retrieved only to be able to use
        # the .count() method, find a more elegant way?
        dist_chunks = self._table_chunks(chrom_table, cols_of_int)
        dist_chunks = [c.groupby("dist_bin").count() for c in dist_chunks]

        # Aggregate them
        # TODO: .add() casts int to float, which should not be an issue
        # (df way smaller than chunks), but maybe check for alternative
        cum_sum = dist_chunks[0]
        for chunk_num in range(1, len(dist_chunks)):
            cum_sum = cum_sum.add(dist_chunks[chunk_num], fill_value=0)
        cum_sum = cum_sum.cumsum()
        del dist_chunks
        print(cum_sum)

        num_chunks = -(-chrom_table.attrs["size"] // self._chunk_size)
        cutoffs = [self._chunk_size * (k + 1) for k in range(num_chunks)]
        cutoffs = [cum_sum[cum_sum["count"] <= c].index[-1] for c in cutoffs]
        cutoffs = [0] + [int(c) for c in cutoffs]
        print(cutoffs)

        # TODO: create arbitrary size chunks or something
        decay_curve = Series(full(len(cum_sum), 0), index=cum_sum.index)
        for ind in range(num_chunks):
            print(f"Decay at beginning of chunk {ind}")
            print(decay_curve)
            container = DataFrame(full((self._chunk_size, 2), 0), columns=cols_of_int)
            lo_bin, hi_bin = cutoffs[ind : ind + 2]
            lo_pos = 0
            for chunk in self._table_chunks(chrom_table, cols_of_int):
                print(chunk)
                index = (chunk["dist_bin"] > lo_bin) & (chunk["dist_bin"] <= hi_bin)
                print(chunk[index])
                hi_pos = lo_pos + sum(index)
                print(lo_pos, hi_pos)
                print(chunk[index].set_index("dist_bin"))
                container[lo_pos:hi_pos] = chunk[index].set_index(
                    "dist_bin", inplace=True
                )
                lo_pos = hi_pos

            print(container.groupby("dist_bin")["count"].median())

    @console_log
    def compute_decay(self, pix_df: DataFrame, stat: str = "median") -> None:
        """Add expected counts ratio column to the pixels dataframe.

        For each pixel compute the expected counts ratio as log2(1 +
        observed/expected), where the expected counts are computed as the
        summary statistic of choice (usually median) of all pixels sharing
        that distance among the two bins composing it.
        """

        pix_df["bin_difference"] = pix_df["bin2_id"] - pix_df["bin1_id"]
        group_counts = pix_df.groupby("bin_difference")["count"]
        pix_df["exp_ratio"] = log2(pix_df["count"] / group_counts.transform(stat) + 1)
        pix_df.drop("bin_difference", axis=1, inplace=True)

    @console_log
    def add_sparsity_val(self, pix_df: DataFrame) -> None:
        """Add alpha value column to dataframe (computed as per Serrano et al. 2009).

        Add a column to the dataframe containing the alpha values for the edges,
        meaning the confidence level of a weighted edge given local fluctuations
        in the network, see Serrano et al. 2009 for in depth explaination. A graph
         is built starting from the list of edges, then the procedure is iterated
        over the nodes; the resulting network is converted back to edge list and
        sorted, since graph traversal does not have inherent order.

        Notes
        -----
        The edge in a disconnected doublet (two nodes connected only to each other by
        a single node) always gets alpha = 1, therefore it will be subsequently filtered.
        In general the number of doublets is very low (at least 7 orders of magnitude
        lower than the total number of edges).
        """

        # Add default alpha value
        pix_df["spar_alpha"] = 1
        graph = from_pandas_edgelist(
            pix_df,
            source="bin1_id",
            target="bin2_id",
            edge_attr=["exp_ratio", "spar_alpha"],
        )

        for node in graph:
            num_neighbours = len(graph[node])

            # The approach is not able to compute an alpha value if the number
            # of neighbours is one, therefore assign the default alpha value
            if num_neighbours == 1:
                continue

            weigths_sum = sum(graph[node][n]["exp_ratio"] for n in graph[node])
            for neigh in graph[node]:
                norm_weight = graph[node][neigh]["exp_ratio"] / weigths_sum
                new_alpha = compute_alpha_val(num_neighbours, norm_weight)
                old_aplha = graph[node][neigh]["spar_alpha"]
                graph[node][neigh]["spar_alpha"] = min(old_aplha, new_alpha)

        graph = to_pandas_edgelist(graph, source="bin1_id", target="bin2_id")

        # Graph object does not preserve pair order
        # TODO: test for performance since it does not feel great
        graph["bin1_id"], graph["bin2_id"] = where(
            graph["bin1_id"] < graph["bin2_id"],
            (graph["bin1_id"], graph["bin2_id"]),
            (graph["bin2_id"], graph["bin1_id"]),
        )

        # TODO: Might be able to remove the sorting step
        graph.sort_values(["bin1_id", "bin2_id"], inplace=True)
        pix_df["spar_alpha"] = graph["spar_alpha"].values

    def _create_chrom_table(
        self,
        chrom_id: str,
        table_root: str,
        remove_dups: bool,
    ) -> None:
        """Create chromosome-level table given the set of parameters."""

        # Retrieve parameters from the parent group
        count_thr = table_root.attrs["count-threshold"]
        dist_thr = table_root.attrs["distance-threshold"]
        decay_stat = table_root.attrs["decay-statistic"]

        # If the table does not already exist
        if chrom_id not in table_root:
            # Create indexer of the rows to keep (optionally flag duplicate pixels)
            ind_to_keep = self._index_filter(chrom_id, count_thr, dist_thr)
            if remove_dups:
                self._deduplicate_index(ind_to_keep, chrom_id)

            # Create table
            table_size = sum(ind_to_keep)
            print(ind_to_keep.shape[0], sum(ind_to_keep))
            chrom_table = self._initialize_table(chrom_id, table_root, table_size)

            # Save chunks to table
            chunks_generator = self._filtered_chunks(chrom_id, ind_to_keep)
            self._save_to_table(chrom_table, chunks_generator)

            # Compute decay statistic
            self._add_bin_dist(chrom_table)
            self._add_norm_dist(chrom_table)

            # # Process the chromosome pixels
            # self.compute_decay(chrom_pix, decay_stat)
            # self.add_sparsity_val(chrom_pix)

        else:
            print(f"WARNING: {chrom_id} already processed with these params, skipping.")

    def _chrom_regex_to_iter(self, chrom_selection):
        """Convert chromosome selection from regex/default string to iterable object."""

        if isinstance(chrom_selection, str):
            if _DEFAULT_CHROM_RE.get(chrom_selection):
                chrom_selection = _DEFAULT_CHROM_RE.get(chrom_selection)
            regex = re.compile(chrom_selection)
            chrom_selection = [c for c in self.chromnames if regex.match(c)]

        return chrom_selection

    def create_tables(
        self,
        chrom_selection: str = "humanCanonical",
        count_thr: int = 0,
        dist_thr: int = 200_000_000,
        decay_stat: str = "median",
        remove_dups: bool = True,
    ) -> None:
        """Process and create chromosome-level tables to use for network construction.

        Given a set of chromosome and some parameters for the processing, create
        individual groups, each one corresponding to a chromosome and containing
        1D-arrays corresponding to the columns of the processed dataframe.

        Parameters
        ----------
        chrom_selection: str or iterable, optional
            Iterable of ids of the chromosome to process or regular expression to build
            one. Some strings are also accepted as proxy for common regular espressions:
            * humanCanonical: "chr1" to "chr22" plus "chrX" and "chrY" (default value)
            * mouseCanonical: "chr1" to "chr19" plus "chrX" and "chrY"
            * others to be defined
        count_thr: int, optional
            Remove pixels whose row count is not greater than this value. (default is 1)
        dist_thr: int, optional
            Remove pixels whose genomic distance among bins is greater or equal to this
            value (in bp). (default is 2Mb)
        decay_stat: str, optional
            Statistic to use to summarize pixels with a certain distance while computing
            the expected counts. Must be compatible with pandas.transform. (default is
            "median")
        """

        # Convert chromosome selection to an iterable
        chrom_selection = self._chrom_regex_to_iter(chrom_selection)

        with open_hdf5(self.store, mode="a") as h5_handle:
            table_root = _GROUP_TEMPLATE.format(count_thr, dist_thr, decay_stat)
            table_root = self.root + "/chrom_tables/" + table_root

            # Create container group if not already existent
            if table_root not in h5_handle:
                table_grp = h5_handle.create_group(table_root)
                table_grp.attrs["count-threshold"] = count_thr
                table_grp.attrs["distance-threshold"] = dist_thr
                table_grp.attrs["decay-statistic"] = decay_stat
            else:
                table_grp = h5_handle[table_root]

            # Create chromosome-level groups and datasets
            for chrom_id in chrom_selection:
                print(f"STARTING {chrom_id}")
                self._create_chrom_table(chrom_id, table_grp, remove_dups)

    def tables_info(self) -> None:
        """Print available chromosome tables for each set of parameters."""

        with open_hdf5(self.store, mode="r") as h5_handle:
            tables_grp = h5_handle[self.root + "/chrom_tables"]
            param_str = "PARAMETER SETS:"
            for param_grp in tables_grp.values():
                param_str += "\n" + "-" * 80
                param_list = [f"\n-{k}: {v}" for k, v in param_grp.attrs.items()]
                param_str += "".join(param_list)
                param_str += "\n-chromosomes:"
                param_str += "".join([f"\n\t--{k}" for k in param_grp.keys()])
            param_str += "\n" + "-" * 80
        print(param_str)

    def available_annotations(self, show: bool = False) -> tuple[str]:
        """Return list of available annotation column names.

        Return a list of all available annotation column names aside from the
        default ones ("chrom", "start", "end") in alphabetical order.
        If show is True, also print them.

        Parameters
        ----------
        show : bool
            If True, print the list of available annotation columns to console.
            (default is True)

        Returns
        -------
        list[str]:
            list of available annotation column names in alphabetical order.
        """

        with open_hdf5(self.store, mode="r") as h5_handle:
            bins_grp = h5_handle[self.root + "/bins"]
            ann_list = tuple(bins_grp.keys())
            ann_list = [k for k in ann_list if k not in ["chrom", "start", "end"]]
            ann_list.sort()

        if show:
            print(f"Available annotation columns: {', '.join(ann_list)}")

        return ann_list

    def tables(
        self,
        chrom_selection: str = "humanCanonical",
        count_thr: int = None,
        dist_thr: int = None,
        decay_stat: str = None,
    ) -> ChromTablesIterator:
        """Return an iterator of selected tables and respective information.

        Use the input parameters to define a list of partial URIs strings
        corresponding to the groups where the column tables are stored, then
        return an iterator object where each item is a tuple in the form
        (corresponding pandas Dataframe, dictionary of dataframe information).

        Parameters
        ----------
        chrom_selection: str or iterable, optional
            Iterable of ids of the chromosome to fetch or regular expression to build
            one. Some strings are also accepted as proxy for common regular espressions:
            * humanCanonical: "chr1" to "chr22" plus "chrX" and "chrY" (default value)
            * other to be defined
        count_thr: int, optional
            Fetch tables created using this value as count threshold. If None, consider
            all tables regardless of the used value.
        dist_thr: int, optional
            Fetch tables created using this value as distance threshold. If None, consider
            all tables regardless of the used value.
        decay_stat: str, optional
            Fetch tables created using this statistic to compute count decay. If None,
            consider all tables regardless of the used statistic.

        Returns
        -------
        :py:class:`ChromTablesIterator` :
            Iterator of tuples in the form (chromsome table, information dictionary).
        """

        # Create valid groups regex according to input parameters
        chrom_selection = self._chrom_regex_to_iter(chrom_selection)
        count_thr = r"\S+" if not count_thr else count_thr
        dist_thr = r"\S+" if not dist_thr else dist_thr
        decay_stat = r"\S+" if not decay_stat else decay_stat
        grp_regex = re.compile(_GROUP_TEMPLATE.format(count_thr, dist_thr, decay_stat))

        # Define a list of partial URIs to valid tables
        with open_hdf5(self.store, mode="r") as h5_handle:
            tables_grp = h5_handle[self.root + "/chrom_tables"]
            valid_groups = [g for g in tables_grp if grp_regex.match(g)]
            valid_tables = []
            for grp in valid_groups:
                tabs = [grp + "/" + c for c in tables_grp[grp] if c in chrom_selection]
                valid_tables.extend(tabs)

        return ChromTablesIterator(self.store, self.root, valid_tables)

    def filter_alpha_tables(
        self,
        tables: ChromTablesIterator,
        alpha_filter: int,
        annot_cols: bool | list[str] = False,
        replace: bool = False,
    ):
        """Return an iterator of filtered (accoring to alpha value) and annotated tables.

        Given an iterator of tables (table, information dictionary), return a new one in
        the same form but with the tables filtered according to some sparsification alpha
        value cutoff (only keep pixels with alpha below the given threshold). If specified,
        annotate the pixels with BED-like coordinates and other informations.

        Parameters
        ----------
        tables : :py:class:`ChromTablesIterator`
            Iterator of chromosome-level tables in the form (table, information dictionary).
        alpha_filter : int
            Sparsification alpha value cutoff (keep pixels with alpha below value).
        annot_cols : bool | list[str] = False
            Columns to use to annotate the pixels in the table. If True annotate only using
            BED-like coordinates columns ("chrom", "start", "end"), if list of strings
            also include those columns.
        replace : False
            If True remove original bin id columns from the table.

        Returns
        -------
        Iterator of (table, information dictionary) tuples.
        """

        for table, info in tables:
            indexer = table[table["spar_alpha"] >= alpha_filter].index
            table.drop(indexer, inplace=True)
            table.drop(["count", "spar_alpha"], axis=1, inplace=True)
            info["alpha-filter"] = alpha_filter

            if annot_cols:
                keep_cols = ["chrom", "start", "end"]
                if isinstance(annot_cols, list):
                    keep_cols.extend(annot_cols)
                lower, upper = self.extent(info["chromosome"])
                bin_table = self.bins()[lower:upper]
                rm_cols = [c for c in bin_table if c not in keep_cols]
                bin_table.drop(rm_cols, axis=1, inplace=True)
                table = annotate(table, bin_table, replace)

            yield (table, info)

    # ////////////////////////////////////////////////////////////////////////
    # ///////////////////////// ANNOTATION FUNCTIONS /////////////////////////
    # ////////////////// Add/process bin annotation columns //////////////////
    # ////////////////////////////////////////////////////////////////////////

    def add_bin_annotation(
        self,
        bed_path: str,
        in_file_col: str = None,
        to_keep_cols: list[str] = None,
    ) -> None:
        """Add bin annotation using a bed-like file.

        Add one or more annotation columns to the bins dataframe. One can add:
        - 0/1 column representing the presence of the bin in the bed-like file
        - any number of columns from the bed-like file (regardless of type)

        Parameters
        ----------
        bed_path : str
            Path to the bed-like file (chrom, start, end, annot1, ..., annotN)
            to use for the annotation.
        in_file_col : str, optional
            If not None, create a 0/1 annotation column (using this string as
            name) with 1 if the bin has (at least) one intersection in the
            bed-like file, 0 otherwise. (default is None)
        to_keep_cols : list[str], optional
            If not None, use the elements of this list as names
            for the annotation columns in the bed-like file to add to the bins
            dataframe. Names are assigned from left to right (ignoring the
            first 3 columns), and any column that receives a name is kept.
            Any column without a name, or to which None was given, is discarded.
            Excess names are ignored. (default is None)

        Notes
        -----
        Adding annotation can drastically increase file size, especially for
        non-numerical annotations; limit categorical annotations (names...).
        """

        # Check for no overlap in old and new annotations
        ann_cols = self.available_annotations()
        if in_file_col:
            if in_file_col in ann_cols:
                raise ValueError("'in file' annotation name already exists.")
        if to_keep_cols:
            if set(ann_cols) & set(to_keep_cols):
                raise ValueError("Overlap of old and new annotations, stopping.")

        # Remove old annotations from the intersection, so the new ones have
        # predictable position (in file: column 5, others: columns 6 and onward)
        bin_bedtool = self.bins()[:].drop(ann_cols, axis=1)

        # NOTE: suppressed error due to pybedtools wrapper implementation
        # pylint: disable=unexpected-keyword-arg, too-many-function-args
        bin_bedtool = BedTool.from_dataframe(bin_bedtool)
        ann_bedtool = BedTool(bed_path)
        bin_bedtool = bin_bedtool.intersect(ann_bedtool, loj=True)
        # pylint: enable=unexpected-keyword-arg, too-many-function-args

        # Convert result to pandas, rename and remove columns
        ann_bins = bin_bedtool.to_dataframe()
        col_names = [None] * ann_bins.shape[1]
        if in_file_col:
            col_names[5] = in_file_col
        if to_keep_cols:
            col_names[6 : 6 + len(to_keep_cols)] = to_keep_cols
        ann_bins.columns = col_names
        ann_bins.drop(labels=[None], axis=1, inplace=True)

        # Free memory space
        del bin_bedtool, ann_bedtool

        # Replace all cells containing only a dot with NaNs, since "." is
        # default for bedtools loj in non-matching non-positional columns
        if in_file_col:
            ann_bins[in_file_col] = [1 if c != -1 else 0 for c in ann_bins[in_file_col]]
        if to_keep_cols:
            ann_bins.replace(r"^\.$", "NaN", regex=True, inplace=True)

        # Create the new bin datasets
        with open_hdf5(self.store, mode="a") as h5_handle:
            grp = h5_handle[self.root + "/bins"]
            for name, vals, dtype in from_df_to_sarrays(ann_bins):
                grp.create_dataset(name, data=vals, dtype=dtype, compression="gzip")
        # TODO: Check cooler.core.put, which should be better for storage efficiency,
        # but currently gives a ValueError for some reason

    def encode_annotation(
        self,
        to_encode: list = None,
        remove_nan_mod: bool = True,
        force_annotation: bool = False,
    ) -> None:
        """Convert bin annotation column(s) to one hot encoding form

        Given a list of bin annotations, replace each of those columns with N
        other columns (where N is the number of modalities for that column)
        each of which in one hot encoding form (1 if modality matches, 0
        otherwise).

        Parameters
        ----------
        to_encode : list, optional
            List (or iterable) of annotations names to split using
            one-hot encoding. The original column is dropped and the derived
            ones are named using {original name}_{modality}. (default is None)
        remove_nan_mod : bool, optional
            Remove nan modality columns resulting from one hot encoding (if any).
            (default is True)
        force_annotation : bool, optional
            Trying to perform one-hot encoding on annotations with many
            modalities will issue and error in order to prevent huge file
            bloating. To suppress the error and split anyway change to True.
            (dafault is False)

        Notes
        -----
        Not found annotation columns are skipped.
        "chrom", "start", "end" columns cannot be removed.
        Removed columns still take space, after this process you might want
        to repack the file (see h5repack tool).
        """

        max_mods = 10  # Critical number of allowed modalities

        # Remove all elements not present in the new column annotations
        to_encode = [ann for ann in to_encode if ann in self.available_annotations()]

        if not to_encode:
            print("WARNING: No valid annotation to encode.")

        # If force, skip modalities number check
        if not force_annotation:
            bins = self.bins()[to_encode]
            too_many_vals = [c for c in to_encode if bins[c][:].nunique() > max_mods]

            if too_many_vals:
                raise ValueError(
                    f"The variable(s) {', '.join(too_many_vals)}"
                    f"has/have more than the maximum number of modalities "
                    f"allowed ({max_mods}). \nThis could lead to "
                    f"massive inflation of the matrix. \nIf you wish to proceed"
                    f" rerun the function with force_annotation=True"
                )

        with open_hdf5(self.store, mode="a") as h5_handle:
            # Create the new bin datasets
            bin_grp = h5_handle[self.root + "/bins"]
            new_bins = get_dummies(self.bins()[to_encode][:], columns=to_encode)
            for name, vals, dtype in from_df_to_sarrays(new_bins):
                bin_grp.create_dataset(name, data=vals, dtype=dtype, compression="gzip")
            # TODO: change to put, see add_bin_annotation

            # Remove those not needed anymore
            if remove_nan_mod:
                to_encode += [c + "_NaN" for c in to_encode if c + "_NaN" in new_bins]
            delete(bin_grp, to_encode)
