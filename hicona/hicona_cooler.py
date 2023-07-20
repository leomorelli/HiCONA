"""Main I/O handling object extending Cooler functionality and format.

Main object to handle .cool/.mcool files and extending them into .hico
format (strict extension, base functionality and structure preserved).

Perform all the pre-processing required to obtain chromosome level
tables which can be stored in the "/chrom_tables" group and retrieved
(as iterator of table matching a criterion).

TODO: Currently loading full chromosome in memory, working on side
branch that performs operations in chunks to limit memory usage.
"""

from collections.abc import Iterable
import re

from cooler import Cooler, create_cooler
from cooler.core import delete
from cooler.util import open_hdf5
from networkx import from_pandas_edgelist, to_pandas_edgelist
import numpy as np
import pandas as pd
from pybedtools import BedTool

from .iterators import ChromTablesIterator, ChunkBordersIterator
from .utils import (
    console_log,
    compute_alpha_val,
    from_df_to_sarrays,
    pd_from_bed,
)

__all__ = ["HiconaCooler"]


# Template to name groups inside chrom_tables group
_GRP_TEMPLATE = "countThr_{}_distThr_{}_stat_{}"
# Maximum number of allowed modalities when transforming an annotation to ohe
_MAX_MODS = 10
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

    # ////////////////////////////////////////////////////////////////////////
    # /////////////////////// PRE-PROCESSING FUNCTIONS ///////////////////////
    # // Functions to pass from full-pixel table to chromosome-level tables //
    # ////////////////////////////////////////////////////////////////////////

    def _create_table(self, grp_path, dataf):
        """Save the dataframe columns as 1D arrays in the specified group"""

        with open_hdf5(self.store, mode="r+") as h5_handle:
            grp = h5_handle[self.root + "/" + grp_path]
            for name, vals, dtype in from_df_to_sarrays(dataf):
                grp.create_dataset(
                    name,
                    data=vals,
                    dtype=dtype,
                    compression="gzip",
                )

    def _pixel_chunks(self, chrom_id: str, size: int):
        """Generator of pixel chunks of specified chromosome and size."""

        extent = self.extent(chrom_id)
        borders = ChunkBordersIterator(self.store, self.root, extent, size)
        for lower, upper in borders:
            yield self.pixels()[lower:upper]

    def _filter_pixels(
        self,
        pix_df: pd.DataFrame,
        chrom_id: str,
        count_thr: int,
        dist_thr: int,
    ) -> pd.DataFrame:
        """Filter out pixels non conformant to some condition.

        Remove all pixels that do NOT satisfy at least one of these filters:
        - distance among the bins is below maximal genomic distance allowed
        - number of counts of the interaction is above the minimal threshold
        - the bins are distinc (non-self looping)
        - the second bin does match the provided chromosome id

        Parameters
        ----------
        pix_df : :py:class:`pd.DataFrame`
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
    def _get_filtered_pix(
        self,
        chrom_id: str,
        count_thr: int,
        dist_thr: int,
        chunk_size: int = 10_000_000,
    ) -> pd.DataFrame:
        """Filter pixels in chunks and return a single dataframe."""

        filt_param = [chrom_id, count_thr, dist_thr]
        c_iter = self._pixel_chunks(chrom_id, size=chunk_size)
        pix_df = [self._filter_pixels(c, *filt_param) for c in c_iter]
        pix_df = pd.concat(pix_df, axis=0)

        return pix_df

    @console_log
    def _drop_duplicate_pixels(self, pix_df: pd.DataFrame) -> pd.DataFrame:
        """pandas.drop_duplicates wrapper for logging purposes."""

        start_size = pix_df.shape[0]
        pix_df.drop_duplicates(subset=["bin1_id", "bin2_id"], inplace=True)
        size_diff = start_size - pix_df.shape[0]
        if size_diff != 0:
            print(f"Warning, {size_diff} duplicate rows were dropped.")

    @console_log
    def _compute_decay(self, pix_df: pd.DataFrame, stat: str) -> None:
        """Add expected counts ratio column to the pixels dataframe.

        For each pixel compute the expected counts ratio as log2(1 +
        observed/expected), where the expected counts are computed as the
        summary statistic of choice (usually median) of all pixels sharing
        that distance among the two bins composing it.
        """

        pix_df["bin_difference"] = pix_df["bin2_id"] - pix_df["bin1_id"]
        grp_df = pix_df.groupby("bin_difference")["count"].transform(stat)
        pix_df["exp_ratio"] = np.log2(pix_df["count"] / grp_df + 1)
        pix_df.drop("bin_difference", axis=1, inplace=True)

    @console_log
    def _add_sparsity_val(self, pix_df: pd.DataFrame) -> None:
        """Add alpha value to dataframe (computed as per Serrano et al. 2009).

        Add a column to the dataframe containing the alpha values for the
        edges, meaning the confidence level of a weighted edge given local
        fluctuations in the network, see Serrano et al. 2009 for in depth
        explanation. A graph is built starting from the list of edges, then
        the procedure is iterated over the nodes; the resulting network is
        converted back to edge list and sorted, since graph traversal does
        not have inherent order.

        Notes
        -----
        The edge in a disconnected doublet (two nodes connected only to each
        other by a single node) always gets alpha = 1, therefore it will be
        subsequently filtered. In general the number of doublets is very low
        (at least 7 orders of magnitude lower than the total number of edges).
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
            num_neigh = len(graph[node])

            # Cannot compute an alpha value if the number of neighbours
            # is one, therefore assign the default alpha value
            if num_neigh == 1:
                continue

            weight_sum = sum(graph[node][n]["exp_ratio"] for n in graph[node])
            for neigh in graph[node]:
                norm_weight = graph[node][neigh]["exp_ratio"] / weight_sum
                new_alpha, _ = compute_alpha_val(num_neigh, norm_weight)
                old_aplha = graph[node][neigh]["spar_alpha"]
                graph[node][neigh]["spar_alpha"] = min(old_aplha, new_alpha)

        # TODO: cannot find if to_pandas_edgelist is already sorted or not
        graph = to_pandas_edgelist(graph, source="bin1_id", target="bin2_id")

        # Graph object does not preserve pair order
        # TODO: test for performance since it does not feel great
        graph["bin1_id"], graph["bin2_id"] = np.where(
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
            chrom_pix = self._get_filtered_pix(chrom_id, count_thr, dist_thr)
            self._drop_duplicate_pixels(chrom_pix)
            self._compute_decay(chrom_pix, decay_stat)
            self._add_sparsity_val(chrom_pix)

            # Save the dataframe columns as individual 1D-arrays
            chrom_table = table_root.create_group(chrom_id)
            for column in chrom_pix.columns:
                chrom_table.create_dataset(column, data=chrom_pix[column])
        else:
            print(f"W: {chrom_id} already processed with these params, skip.")

    def _chrom_regex_to_iter(self, chrom_selection):
        """Convert chromosome selection from regex/default str to iterable."""

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
    ) -> None:
        """Create chromosome-level tables to use for network construction.

        Given a set of chromosome and some parameters for the processing,
        create individual groups, each one corresponding to a chromosome
        and containing 1D-arrays corresponding to the columns of the
        processed dataframe.

        Parameters
        ----------
        chrom_selection: str or iterable, optional
            Iterable of ids of the chromosome to process or regular expression
            to build one. Some strings are also accepted as proxy for common
            regular espressions:
            * humanCanonical: "chr1" to "chr22" plus "chrX" and "chrY"
            * mouseCanonical: "chr1" to "chr19" plus "chrX" and "chrY"
            * others to be defined
            (default is "humanCanonical")
        count_thr: int, optional
            Remove pixels whose row count is not greater than this value.
            (default is 1)
        dist_thr: int, optional
            Remove pixels whose genomic distance among bins is greater or
            equal to this value (in bp). (default is 2Mb)
        decay_stat: str, optional
            Statistic used to summarize pixels with a certain distance while
            computing the expected counts. Must be compatible with
            pandas.transform. (default is "median")
        """

        # Convert chromosome selection to an iterable
        chrom_selection = self._chrom_regex_to_iter(chrom_selection)

        with open_hdf5(self.store, mode="a") as h5_handle:
            table_root = _GRP_TEMPLATE.format(count_thr, dist_thr, decay_stat)
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
            par_str = "PARAMETER SETS:"
            for par_grp in tables_grp.values():
                par_str += "\n" + "-" * 80
                par_lst = [f"\n-{k}: {v}" for k, v in par_grp.attrs.items()]
                par_str += "".join(par_lst)
                par_str += "\n-chromosomes:"
                par_str += "".join([f"\n\t--{k}" for k in par_grp.keys()])
            par_str += "\n" + "-" * 80
        print(par_str)

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
            Iterable of ids of the chromosome to fetch or regular expression
            to build one. Some strings are also accepted as proxy for common
            regular espressions:
            * humanCanonical: "chr1" to "chr22" plus "chrX" and "chrY"
            * mouseCanonical: "chr1" to "chr19" plus "chrX" and "chrY"
            * other to be defined
            (default is "humanCanonical")
        count_thr: int, optional
            Fetch tables created using this value as count threshold.
            If None, consider all tables regardless of the used value.
        dist_thr: int, optional
            Fetch tables created using this value as distance threshold.
            If None, consider all tables regardless of the used value.
        decay_stat: str, optional
            Fetch tables created using this statistic to compute count decay.
            If None, consider all tables regardless of the used statistic.

        Returns
        -------
        :py:class:`ChromTablesIterator` :
            Iterator of tuples in the form (chrom table, information dict).
        """

        # Create valid groups regex according to input parameters
        chroms = self._chrom_regex_to_iter(chrom_selection)
        count_thr = r"\S+" if not count_thr else count_thr
        dist_thr = r"\S+" if not dist_thr else dist_thr
        decay_stat = r"\S+" if not decay_stat else decay_stat
        filt_stats = [count_thr, dist_thr, decay_stat]
        grp_regex = re.compile(_GRP_TEMPLATE.format(*filt_stats))

        # Define a list of partial URIs to valid tables
        with open_hdf5(self.store, mode="r") as h5_handle:
            tables_grp = h5_handle[self.root + "/chrom_tables"]
            valid_groups = [g for g in tables_grp if grp_regex.match(g)]
            valid_tables = []
            for grp in valid_groups:
                tabs = [grp + "/" + c for c in chroms if c in tables_grp[grp]]
                valid_tables.extend(tabs)

        return ChromTablesIterator(self.store, self.root, valid_tables)

    # ////////////////////////////////////////////////////////////////////////
    # ///////////////////////// ANNOTATION FUNCTIONS /////////////////////////
    # ////////////////// Add/process bin annotation columns //////////////////
    # ////////////////////////////////////////////////////////////////////////

    def _valid_bin_annotations(self, names: str | Iterable[str]):
        """Return only valid bin annotation names as iterable of strings"""

        names = [] if not names else names
        names = [names] if isinstance(names, str) else names
        names = [n for n in names if n in self.annotations_list()]

        return names

    def annotations_list(self) -> list[str]:
        """Return a list of available bin annotation columns

        Return a list of all available bin annotation columns (that is, all
        columns in the bins group besides "chrom", "start", and "end") sorted
        alphabetically.

        Returns
        -------
        ann_list : list[str]
        """

        with open_hdf5(self.store, mode="r") as h5_handle:
            bins_grp = h5_handle[self.root + "/bins"]
            ann_list = tuple(bins_grp.keys())
        ann_list = [k for k in ann_list if k not in ["chrom", "start", "end"]]
        ann_list.sort()

        return ann_list

    def add_bin_annotation(
        self,
        bed_path: str,
        in_file: str = None,
        to_keep: Iterable[str | None] = None,
    ) -> None:
        """Add bin annotation(s) using a bed-like file.

        Add one or more annotation columns to the bins dataframe. One can add:
        - 0/1 column representing an overlap of the bin in the bed-like file
        - any number of columns from the bed-like file (regardless of type)

        Parameters
        ----------
        bed_path : str
            Path to the bed-like file (chrom, start, end, annot1, ..., annotN)
            to use for the annotation.
        in_file : str, optional
            Name of the 0/1 annotation column, containing 1 if the bin has at
            least one overlap with any interval in the bed-file, 0 otherwise.
            If None, no such column is created. (default is None)
        to_keep : Iterable[str], optional
            Names for the columns of the bed-like file to add to the bins
            dataframe. Names are assigned from left to right (ignoring the
            first 3 columns), and any column that receives a name is kept.
            Any column without a name is discarded. To skip a column place a
            None in its position. Excess names are ignored. (default is None)

        Notes
        -----
        Adding annotation can drastically increase file size, especially for
        non-numerical annotations; limit categorical annotations with many
        distinct values (names, ids, ...).
        """

        def _intersect_dataframes(df_a, df_b):
            """Return dataframe intersection using bedtools intersect -loj"""

            # Suppress linting error due to pybedtools wrapper implementation
            # pylint: disable=unexpected-keyword-arg, too-many-function-args
            bed_a = BedTool.from_dataframe(df_a)
            bed_b = BedTool.from_dataframe(df_b)
            bed_a = bed_a.intersect(bed_b, loj=True)
            # pylint: enable=unexpected-keyword-arg, too-many-function-args
            return bed_a.to_dataframe()

        # Convert to_keep to None if all elements are None
        to_keep = to_keep if any(to_keep) else None

        # Check for no overlap in old and new annotations
        if in_file in self.annotations_list():
            raise ValueError("'in file' annotation name already exists.")
        if to_keep:
            if set(to_keep) & set(self.annotations_list()):
                raise ValueError("Overlap with old annotations, stopping.")

        # Create the two bin df and merge on default bed columns
        bin_df = self.bins()[["chrom", "start", "end"]][:]
        ann_df = pd_from_bed(bed_path)
        bin_df = _intersect_dataframes(bin_df, ann_df.iloc[:, :3])

        # Create OHE column where 1 = "intersection with annotation"
        if in_file:
            in_file_df = [1 if c != -1 else 0 for c in bin_df.iloc[:, 5]]
            in_file_df = pd.DataFrame(in_file_df, columns=[in_file])
            self._create_table("bins", in_file_df)

        # Crate any other specified annotation columns
        if to_keep:
            # Bedtools automatically assigns these values
            to_intersect_on = ["name", "score", "strand"]

            # Rename columns for merge and pad with None
            ann_cols = to_intersect_on + list(to_keep)
            if (pad := len(ann_df.columns) - len(ann_cols)) > 0:
                ann_cols = ann_cols + [None] * pad
            ann_df.columns = ann_cols

            # Merge and save only required columns
            ann_df = bin_df.merge(ann_df, how="left", on=to_intersect_on)
            ann_df.drop(labels=[None], axis=1, inplace=True)
            self._create_table("bins", ann_df.iloc[:, 6:])
            # Columns 0-5 are "chrom" "start" "end" "name" "score" "strand"

    def del_bin_annotation(self, to_del: str | Iterable[str]) -> None:
        """Remove bin annotation columns.

        Given one or more bin annotation names, remove those columns from the
        bins table. Non-existent annotations or default bin columns ("chrom",
        "start", "end") are skipped without raising warning/errors. Removed
        columns still take space, after this process you might want to repack
        the file (see h5repack tool).

        Parameters
        ----------
        to_del : str | Iterable[str]
            String or iterable of them representing bin annotations to remove.
        """

        to_del = self._valid_bin_annotations(to_del)
        with open_hdf5(self.store, mode="r+") as h5_handle:
            bin_grp = h5_handle[self.root + "/bins"]
            delete(bin_grp, to_del)

    def annotation_to_ohe(
        self,
        to_ohe: Iterable[str],
        remove_original: bool = False,
        remove_nan_mod: bool = True,
        force_annotation: bool = False,
    ) -> None:
        """Convert bin annotation column(s) to one hot encoding form

        Given a list of bin annotations, create for each of those columns N
        other columns (where N is the number of modalities, or unique values,
        for that column) each of which in one hot encoding form (1 if modality
        matches, 0 otherwise). If specified, remove the original column.

        Parameters
        ----------
        to_ohe : Iterable[str]
            Iterable of annotations names to perform one-hot encoding on.
            Generated columns are named using {original name}_{modality}.
        remove_original : bool, optional
            Whether to remove the original annotation columns on which ohe is
            performed on. (default is False)
        remove_nan_mod : bool, optional
            Whether to remove columns originated from ohe of the NaN modality,
            meaning {original name}_NaN, if any. (default is True)
        force_annotation : bool, optional
            Force ohe even though the number of modalities of one or more
            columns would exceed the default maximum, potentially leading to
            a huge file size increase. (default is False)

        Notes
        -----
        Non-existent annotations are skipped without raising warning/errors.
        Removed columns still take space, after this process you might want
        to repack the file (see h5repack tool).
        """

        # Select and retrieve needed annotation columns
        to_ohe = self._valid_bin_annotations(to_ohe)
        ann_df = self.bins()[to_ohe][:]

        # If force, skip modalities number check
        if not force_annotation:
            too_many = [c for c in to_ohe if ann_df[c].nunique() > _MAX_MODS]
            if too_many:
                raise ValueError(
                    f"The variable(s) {', '.join(too_many)} has/have more "
                    f"than the default max number of modalities ({_MAX_MODS})"
                    f".\nThis could lead to a huge file size increase. To "
                    f"proceed anyway, rerun with force_annotation=True."
                )

        # Generate ohe df and save to h5
        ohe_df = pd.get_dummies(ann_df, columns=to_ohe)
        if remove_nan_mod:
            ohe_df = ohe_df[[c for c in ohe_df if not c.endswith("_NaN")]]
        self._create_table("bins", ohe_df)

        # Remove original columns if selected
        if remove_original:
            self.del_bin_annotation(to_ohe)

    # ////////////////////////////////////////////////////////////////////////
    # ////////////////////// MCOOL GENERATION FUNCTIONS //////////////////////
    # //////////////////// Create new .cool/.mcool files  ////////////////////
    # ////////////////////////////////////////////////////////////////////////

    def gen_sparsified_cooler(
        self,
        cool_uri: str,
        chr_tables: ChromTablesIterator,
        alpha_thr: str | float | Iterable[float],
    ):
        """Create a cool/mcool file containing only sparsified pixels.

        Placeholder
        """
        # TODO: Add alpha lenght check
        # TODO: Check the same chromosome was not given twice

        def _filter_alpha_tables(tables, alphas):
            """Filter iterable of tables according to an iterable of alphas"""
            for (table, _), alpha in zip(tables, alphas):
                table = table[table["spar_alpha"] < alpha]
                yield table[["bin1_id", "bin2_id", "count"]]

        if isinstance(alpha_thr, str):
            pass  # TODO: compute optimal values
        elif isinstance(alpha_thr, float):
            alpha_thr = [alpha_thr] * len(chr_tables)

        bare_bins = self.bins()[["chrom", "start", "end"]][:]
        filt_pix = _filter_alpha_tables(chr_tables, alpha_thr)

        create_cooler(mcool_uri, bins=bare_bins, pixels=filt_pix)
