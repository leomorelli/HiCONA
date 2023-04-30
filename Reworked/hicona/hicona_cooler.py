"""Placeholder
Placeholder
"""

import re

from cooler import Cooler, annotate
from cooler.core import delete
from cooler.util import open_hdf5
from networkx import from_pandas_edgelist, to_pandas_edgelist
from numpy import log2, where
from pandas import concat, DataFrame, get_dummies
from pybedtools import BedTool

from .iterators import ChunkBordersIterator, ChromTablesIterator
from .utils import console_log, compute_alpha_val, from_df_to_sarrays

__all__ = ["HiconaCooler"]


# Template to name groups inside chrom_tables group
_GROUP_TEMPLATE = "countThr_{}_distThr_{}_stat_{}"
# Dictionary of standard regular expressions to simplify chromosome fetching
_DEFAULT_CHROM_RE = {
    "humanCanonical": "^chr([1-9]|[1][0-9]|[2][0-2]|[XY])$",
    "mouseCanonical": "^chr([1-9]|[1][0-9]|[XY])$",
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

    def _pixel_chunks(self, chrom_id: str, chunk_size: int):
        """Generator function for pixel chunks of specified chromosome and size."""

        extent = self.extent(chrom_id)
        borders = ChunkBordersIterator(self.store, self.root, extent, chunk_size)
        for lower, upper in borders:
            yield self.pixels()[lower:upper]

    def filter_pixels(
        self,
        pix_df: DataFrame,
        chrom_id: str,
        count_thr: int = 1,
        dist_thr: int = 10_000_000,
    ) -> None:
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
        count_thr : int, optional
            Threshold for pixel count; only keep pixels with count greater
            than this value. (default is 1)
        dist_thr : int, optional
            Threshold for genomic distance; only keep pixels whose bins are
            closer to each other than this distance.

        Notes
        -----
        It is assumed that the table to filter was obtained through
        :py:meth:`Cool.pixels.fetch` and therefore that a) ``bin1_id`` field
        always matches the chromosome of interest, b) there is no ``bin2_id``
        lower than the smallest possible bin id for the specified chromosome.
        """

        max_diff = -(-dist_thr // self.binsize)
        id_upper_bound = self.extent(chrom_id)[1]

        indexer = pix_df[
            (pix_df["bin2_id"] - pix_df["bin1_id"] >= max_diff)
            | (pix_df["count"] <= count_thr)
            | (pix_df["bin1_id"] == pix_df["bin2_id"])
            | (pix_df["bin2_id"] >= id_upper_bound)
        ].index
        pix_df.drop(indexer, inplace=True)

        return pix_df

    @console_log
    def get_filtered_pixels(self, chrom_id, count_thr, dist_thr, chunk_size=10_000_000):
        """Filter pixels in chunks and return a single dataframe."""

        c_iter = self._pixel_chunks(chrom_id, chunk_size=chunk_size)
        pix_df = [self.filter_pixels(c, chrom_id, count_thr, dist_thr) for c in c_iter]
        pix_df = concat(pix_df, axis=0)

        return pix_df

    @console_log
    def drop_duplicate_pixels(self, pix_df):
        """pandas.drop_duplicates wrapper for logging purposes."""
        pix_df.drop_duplicates(subset=["bin1_id", "bin2_id"], inplace=True)

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

        # TODO: cannot find if to_pandas_edgelist is already sorted or not
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

    def _create_chrom_table(self, chrom_id: str, table_root: str) -> None:
        """Create chromosome-level table given the set of parameters."""

        # Retrieve parameters from the parent group
        count_thr = table_root.attrs["count-threshold"]
        dist_thr = table_root.attrs["distance-threshold"]
        decay_stat = table_root.attrs["decay-statistic"]

        if chrom_id not in table_root:
            # Process the chromosome pixels
            chrom_pix = self.get_filtered_pixels(chrom_id, count_thr, dist_thr)
            self.drop_duplicate_pixels(chrom_pix)
            self.compute_decay(chrom_pix, decay_stat)
            self.add_sparsity_val(chrom_pix)

            # Save the dataframe columns as individual 1D-arrays (for storage)
            chrom_table = table_root.create_group(chrom_id)
            for column in chrom_pix.columns:
                chrom_table.create_dataset(column, data=chrom_pix[column])
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
        count_thr: int = 1,
        dist_thr: int = 2_000_000,
        decay_stat: str = "median",
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
                self._create_chrom_table(chrom_id, table_grp)

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
