"""Main I/O handling object extending Cooler functionality and format.

Main object to handle .cool/.mcool files in order to perform all the
pre-processing required to obtain chromosome level tables.
Chromosome-level tables are stored in the ``/chrom_tables`` group and
can be retrieved to create filtered networks to analyze.
"""

from collections.abc import Iterable
import re
from statistics import median

from cooler import Cooler, create_cooler
from cooler.core import delete
import h5py
import numpy as np
import pandas as pd
from pybedtools import BedTool
from scipy import integrate

import matplotlib.pyplot as plt
from .iterators import ChromTablesIterator
from .utils import (
    console_log,
    from_df_to_sarrays,
    get_chunk_borders,
    pd_from_bed,
    round_half_up,
)

__all__ = ["HiconaCooler"]


# Template to name groups inside chrom_tables group
_GRP_TEMPLATE = "distThr_{}_countThr_{}"
# Maximum number of allowed modalities when transforming an annotation to ohe
_MAX_MODS = 10
# Minimum number of pixels per table chunk
_MIN_PIX_CHUNK = 1_000_000
# Dictionary of standard regular expressions to simplify chromosome fetching
_DEFAULT_CHROM_RE = {
    "humanCanonical": "^chr([1-9]|[1][0-9]|[2][0-2]|[XY])$",
    "mouseCanonical": "^chr([1-9]|[1][0-9]|[XY])$",
}


class HiconaCooler(Cooler):
    """An extension of the Cooler class to prepare data for network analysis.

    :py:class:`HiconaCooler` inherits from :py:class:`cooler.Cooler` and
    extends it by adding new functionalities, mainly revolving around
    the creation of chromosome-level tables to use for network analyses.
    Tables are stored in a separate group (``chrom_tables``) of the
    :py:class:`h5py.File` and no method or property of the :py:class:`Cooler`
    is overwritten, therefore a :py:class:`HiconaCooler` object can always be
    used as a :py:class:`Cooler` one.

    Parameters
    ----------
    store : str, :py:class:`h5py.File` or :py:class:`h5py.Group`
        Path to a cooler file, URI string, or open handle to the root HDF5
        group of a cooler data collection.
    kwargs : optional
        Options to be passed to :py:class:`h5py.File` upon every access.
        See class constructor for :py:class:`Cooler` for more detail.
    """

    # ////////////////////////////////////////////////////////////////////////
    # //////////////////////// BASIC OBJECT FUNCTIONS ////////////////////////
    # /////// Class constructor, setter, getters and similar functions ///////
    # ////////////////////////////////////////////////////////////////////////

    # TODO: autodetect system resources to define better chunk size

    def __init__(self, store: str | h5py.File | h5py.Group, **kwargs):
        # Mask deprecated root parameter from super-class
        super().__init__(store, **kwargs)
        self._chunk_size = 1_000_000

    @property
    def chunk_size(self):
        """Size of fixed lenght chunks used during processing."""
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, value):
        if not (isinstance(value, int)) or value < _MIN_PIX_CHUNK:
            raise ValueError(f"chunk_size must be: int >= {_MIN_PIX_CHUNK}.")
        self._chunk_size = value

    # ////////////////////////////////////////////////////////////////////////
    # //////////////////////////// I/O FUNCTIONS /////////////////////////////
    # ////////// Functions to create or retrieve tables and groups ///////////
    # ////////////////////////////////////////////////////////////////////////

    def _create_table(self, grp_path, dataf):
        """Save the dataframe columns as 1D arrays in the specified group."""

        with h5py.File(self.store, mode="r+") as h5_handle:
            grp = h5_handle[self.root + "/" + grp_path]
            for name, vals, dtype in from_df_to_sarrays(dataf):
                grp.create_dataset(
                    name,
                    data=vals,
                    dtype=dtype,
                    compression="gzip",
                )

    def _chrom_chunks(self, chrom_id):
        """Return chunk borders pairs for the pixels of a chromosome"""

        extent = self.extent(chrom_id)
        with h5py.File(self.store, mode="r") as h5_handle:
            h5_grp = h5_handle[self.root]
            min_off = h5_grp["indexes/bin1_offset"][extent[0]]
            max_off = h5_grp["indexes/bin1_offset"][extent[1]]

        return get_chunk_borders(min_off, max_off, self._chunk_size)

    def _table_chunks(self, table_uri):
        """Return chunk borders pairs for a pixel table."""

        with h5py.File(self.store, mode="r") as h5_handle:
            table = h5_handle[table_uri]
            max_off = table.attrs["num_pixels"]

        return get_chunk_borders(0, max_off, self._chunk_size)

    def _put_chunk(self, chunk, table_uri, bounds, cols=None):
        """Place pixel chunk in table at the given position."""
        lower, upper = bounds
        cols = cols if cols else list(chunk)
        with h5py.File(self.store, mode="r+") as h5_handle:
            table = h5_handle[table_uri]
            for col in cols:
                table[col][lower:upper] = chunk[col]

    def _get_chunk(self, table_uri, bounds, cols=None):
        """Retrieve a chunk of pixels from a specified table."""
        lower, upper = bounds
        with h5py.File(self.store, mode="r") as h5_handle:
            table = h5_handle[table_uri]
            cols = cols if cols else list(table.keys())
            chunk = pd.DataFrame({f: table[f][lower:upper] for f in cols})
        return chunk

    # ////////////////////////////////////////////////////////////////////////
    # /////////////////////// PRE-PROCESSING FUNCTIONS ///////////////////////
    # // Functions to pass from full-pixel table to chromosome-level tables //
    # ////////////////////////////////////////////////////////////////////////

    def _filter_chunk(self, pix_df, chrom_id, count_thr, dist_thr):
        """
        Filter out pixels which are either self-looping, inter-chromosomal,
        above/equal to distance threshold, below/equal to count thrshold.
        Current version does not work if inter-chromosomal pixels are kept.
        """

        max_diff = -(-dist_thr // self.binsize)
        id_upper_bound = self.extent(chrom_id)[1]

        filtered_df = pix_df[
            (pix_df["bin2_id"] - pix_df["bin1_id"] < max_diff)
            & (pix_df["count"] > count_thr)
            & (pix_df["bin1_id"] != pix_df["bin2_id"])
            & (pix_df["bin2_id"] < id_upper_bound)
        ]

        return filtered_df

    def _get_table_size(self, chrom_id, filt_opts):
        """Determine table size by running mock filtering."""

        size = 0
        for bounds in self._chrom_chunks(chrom_id):
            chunk = self._get_chunk(f"{self.root}/pixels", bounds)
            size += len(self._filter_chunk(chunk, chrom_id, **filt_opts))
        return size

    def _init_table(self, table_root, chrom_id, filt_opts):
        """Initialize chromosome-level pixel table to fill in."""

        size = self._get_table_size(chrom_id, filt_opts)
        with h5py.File(self.store, mode="r+") as h5_handle:
            table_group = h5_handle[table_root]
            chrom_table = table_group.require_group(chrom_id)
            chrom_table.require_dataset("bin1_id", (size,), dtype="i8")
            chrom_table.require_dataset("bin2_id", (size,), dtype="i8")
            chrom_table.require_dataset("count", (size,), dtype="i4")
            chrom_table.require_dataset("exp_ratio", (size,), dtype="f8")
            chrom_table.require_dataset("alpha_min", (size,), dtype="f8")
            chrom_table.require_dataset("alpha_max", (size,), dtype="f8")

            chrom_table.attrs["chromosome"] = chrom_id
            chrom_table.attrs["num_pixels"] = size

    def _filter_pixels(self, table_uri, chrom_id, filt_opts):
        """Filter and store pixels in the chromosome-table."""

        lower = 0
        for bounds in self._chrom_chunks(chrom_id):
            chunk = self._get_chunk(f"{self.root}/pixels", bounds)
            chunk = self._filter_chunk(chunk, chrom_id, **filt_opts)
            upper = lower + len(chunk)
            self._put_chunk(chunk, table_uri, (lower, upper))
            lower = upper

    def _norm_curve(self, table_uri):
        """Compute curve for genomic distance normalization."""

        chunk_cols = ["bin1_id", "bin2_id", "count"]
        curve = pd.Series()
        for bounds in self._table_chunks(table_uri):
            chunk = self._get_chunk(table_uri, bounds, cols=chunk_cols)
            chunk["bin_diff"] = chunk["bin2_id"] - chunk["bin1_id"]
            vals = chunk.groupby("bin_diff")["count"].apply(list)
            curve = curve.combine(vals, lambda x, y: x + y, fill_value=[])

        return curve.apply(median)

    def _plot_norm_curve(self, norm_curve):
        """Temporary function to plot normalization curve."""
        # TODO: Remove or move elsewhere

        norm_curve.plot()
        plt.yscale("log")
        plt.xscale("log")
        plt.show()

    def _normalize_pixels(self, table_uri, norm_curve):
        """Normalize chromosome-level table for genomic distance."""

        chunk_cols = ["bin1_id", "bin2_id", "count"]
        lower = 0
        for bounds in self._table_chunks(table_uri):
            chunk = self._get_chunk(table_uri, bounds, cols=chunk_cols)

            # Compute expected ratios
            chunk["bin_diff"] = chunk["bin2_id"] - chunk["bin1_id"]
            chunk = chunk.join(norm_curve.rename("dist_norm"), on="bin_diff")
            exp_ratios = np.log2(chunk["count"] / chunk["dist_norm"] + 1)
            chunk["exp_ratio"] = exp_ratios

            # Save expected ratios
            upper = lower + len(exp_ratios)
            self._put_chunk(chunk, table_uri, (lower, upper), ("exp_ratio",))
            lower = upper

    def _get_node_stats(self, table_uri):
        """Compute sum of weights and degree for each node/bin."""

        # Initialize empty containers
        weights = pd.Series()
        degrees = pd.Series()

        chunk_cols = ["bin1_id", "bin2_id", "exp_ratio"]
        for bounds in self._table_chunks(table_uri):
            chunk = self._get_chunk(table_uri, bounds, cols=chunk_cols)
            for bin_col in ["bin1_id", "bin2_id"]:
                # Compute metrics on chunk
                grouped = chunk[[bin_col, "exp_ratio"]].groupby(bin_col)
                chunk_weights = grouped.sum()["exp_ratio"]
                chunk_degrees = grouped.count()["exp_ratio"]

                # Increase counters
                weights = weights.add(chunk_weights, fill_value=0)
                degrees = degrees.add(chunk_degrees, fill_value=0)

        return pd.DataFrame({"weight": weights, "degree": degrees})

    @console_log
    def _add_spar_alpha(self, table_uri, node_stats):
        """Compute alpha value as per Serrano et al. 2009."""

        def compute_alpha(row):
            """Given a (weight, degree) pair, compute the integral."""

            weight, deg = row["norm_weight"], row["degree"]
            res, _ = integrate.quad(lambda x: (1 - x) ** (deg - 2), 0, weight)
            alpha = 1 - (deg - 1) * res

            return round_half_up(alpha, 4)

        def unique_alphas(dataf):
            """Return alpha values of unique (norm_weight, deg) pairs."""

            values = dataf[["degree", "norm_weight"]].drop_duplicates()
            values[f"alpha_{num}"] = 1
            mask = values["degree"] != 1
            alphas = values.loc[mask].apply(compute_alpha, axis=1)
            values.loc[mask, f"alpha_{num}"] = alphas

            return values

        # For each chunk of the chromosome-level pixel table
        chunk_cols = ["bin1_id", "bin2_id", "exp_ratio"]
        for bounds in self._table_chunks(table_uri):
            chunk = self._get_chunk(table_uri, bounds, cols=chunk_cols)

            # For both bins composing the pixel
            for num, bin_col in enumerate(["bin1_id", "bin2_id"]):
                # Add node statistics and normalized weight for that bin
                chunk = chunk.merge(
                    node_stats,
                    how="left",
                    left_on=bin_col,
                    right_index=True,
                )
                chunk["norm_weight"] = chunk["exp_ratio"] / chunk["weight"]

                # Compute and add the alpha values for each row
                chunk = chunk.merge(
                    unique_alphas(chunk),
                    how="left",
                    on=["degree", "norm_weight"],
                )

                # Remove node specific information
                tmp_cols = ["weight", "degree", "norm_weight"]
                chunk.drop(tmp_cols, axis=1, inplace=True)

            # Sort the values into min and max column, then remove tmp ones
            chunk["alpha_min"] = chunk[["alpha_0", "alpha_1"]].min(axis=1)
            chunk["alpha_max"] = chunk[["alpha_0", "alpha_1"]].max(axis=1)
            chunk.drop(["alpha_0", "alpha_1"], axis=1, inplace=True)

            cols_to_store = ["alpha_min", "alpha_max"]
            self._put_chunk(chunk, table_uri, bounds, cols=cols_to_store)

    def _create_chrom_table(self, chrom_id, table_root, filt_opts):
        """Create chromosome-level table given the set of parameters."""

        def does_not_exist(chrom_id, store, table_root):
            """Check whether chromosome was already processed."""
            with h5py.File(store, mode="r") as h5_handle:
                table = h5_handle[table_root]
            return chrom_id in table

        # if does_not_exist(chrom_id, self.store, table_root):
        if True:
            table_uri = table_root + "/" + chrom_id
            self._init_table(table_root, chrom_id, filt_opts)
            self._filter_pixels(table_uri, chrom_id, filt_opts)

            norm_curve = self._norm_curve(table_uri)
            # self._plot_norm_curve(norm_curve)
            self._normalize_pixels(table_uri, norm_curve)

            node_stats = self._get_node_stats(table_uri)
            self._add_spar_alpha(table_uri, node_stats)
            quit()

            for chunk in self._table_chunks(table_uri):
                print(chunk)

            # Save the dataframe columns as individual 1D-arrays
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

    def _init_tables_grp(self, dist_thr, count_thr):
        """Initialize main table group and param specific group if needed."""

        # TODO: remove
        print(f"dist_thr: {dist_thr}, count_thr: {count_thr}")

        with h5py.File(self.store, mode="r+") as h5_handle:
            table_root = _GRP_TEMPLATE.format(dist_thr, count_thr)
            table_root = self.root + "/chrom_tables/" + table_root

            # Create container group if not already existent
            if table_root not in h5_handle:
                table_grp = h5_handle.create_group(table_root)
                table_grp.attrs["count-threshold"] = count_thr
                table_grp.attrs["distance-threshold"] = dist_thr

        return table_root

    def create_tables(
        self,
        chrom_selection: str | Iterable[str] = "humanCanonical",
        dist_thr: int = 200_000_000,
        count_thr: int = 0,
    ) -> None:
        """Create chromosome-level tables to use for network construction.

        Given a set of chromosomes and some processing parameters, create
        individual h5-file groups, each one corresponding to a chromosome
        and containing 1D-arrays corresponding to the columns of the
        processed dataframe.

        The steps performed to create the tables from the starting bins are:

        - **Filtering**: remove inter-chromosomal pixels, self-looping pixels
          (``bin1_id == bin2_id``), pixels with a count below ``count_thr``,
          pixels with a genomic distance among the bins greater than
          ``dist_thr``.
        - **Computing decay**: compute counts normalized for the fact that
          genomically closer bins have a higher probability of random contact
          (therefore higher counts by chance). That is
          :math:`normCount = log_2(rawCount/(normFactor + 1))`, where
          :math:`normFactor` is the ``decay_stat`` applied on the set of all
          pixels with the same genomic distance as the one being normalized.
        - **Sparsification**: compute the sparsification score of each pixel
          (using normalized counts) according to ``Serrano et al. 2009``.

        Parameters
        ----------
        chrom_selection: str or Iterable[str], optional
            Iterable of chromosome ids to process or regular expression.
            Some strings are also accepted as proxy for common selections:

            - ``humanCanonical``: "chr1" to "chr22" plus "chrX" and "chrY"
            - ``mouseCanonical``: "chr1" to "chr19" plus "chrX" and "chrY"
            - others to be defined

            (default is ``humanCanonical``)
        dist_thr: int, optional
            Remove pixels whose genomic distance among bins is greater or
            equal to this value (in bp). (default is 2Mb)
        count_thr: int, optional
            Remove pixels whose row count is not greater than this value.
            (default is 0)
        """

        # TODO: probably better to set the default to all chromosome in file

        # Initialize table container
        filt_opts = {"dist_thr": dist_thr, "count_thr": count_thr}
        table_root = self._init_tables_grp(**filt_opts)

        # Create chromosome-level groups and datasets
        for chrom_id in self._chrom_regex_to_iter(chrom_selection):
            print(f"STARTING to work on: {chrom_id}")
            self._create_chrom_table(chrom_id, table_root, filt_opts)

    def list_tables(self) -> None:
        """Print available chromosome tables for each set of parameters."""

        # TODO: Maybe find a prettier and more flexible way to print
        with h5py.File(self.store, mode="r") as h5_handle:
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

        Use the input parameters to define which tables to retrieve, then
        return a :py:class:`ChromTablesIterator` where each item is a tuple in
        the form ``(DataFrame, dict)``.

        Parameters
        ----------
        chrom_selection: str or Iterable[str], optional
            Iterable of chromosome ids to retrieve or regular expression.
            Some strings are also accepted as proxy for common selections:

            - ``humanCanonical``: "chr1" to "chr22" plus "chrX" and "chrY"
            - ``mouseCanonical``: "chr1" to "chr19" plus "chrX" and "chrY"
            - others to be defined

            (default is ``humanCanonical``)
        count_thr: int, optional
            Fetch tables created using this value as count threshold.
            If None, get all tables regardless of the used value.
        dist_thr: int, optional
            Fetch tables created using this value as distance threshold.
            If None, get all tables regardless of the used value.
        decay_stat: str, optional
            Fetch tables created using this statistic for count decay.
            If None, get all tables regardless of the used statistic.

        Returns
        -------
        :py:class:`ChromTablesIterator` :
            Iterator of tuples in the form ``(chrom table, info dict)``.
        """

        # Create valid groups regex according to input parameters
        chroms = self._chrom_regex_to_iter(chrom_selection)
        count_thr = r"\S+" if not count_thr else count_thr
        dist_thr = r"\S+" if not dist_thr else dist_thr
        decay_stat = r"\S+" if not decay_stat else decay_stat
        filt_stats = [count_thr, dist_thr, decay_stat]
        grp_regex = re.compile(_GRP_TEMPLATE.format(*filt_stats))

        # Define a list of partial URIs to valid tables
        with h5py.File(self.store, mode="r") as h5_handle:
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
        names = [n for n in names if n in self.list_annotations()]

        return names

    def list_annotations(self) -> Iterable[str]:
        """Return an iterable of available bin annotation columns.

        Return an iterable of all available bin annotation columns (that is,
        all columns in the bins group besides ``chrom``, ``start``, and
        ``end``) sorted alphabetically.

        Returns
        -------
        Iterable[str] :
            Iterable of bin annotation names in alphabetical order.
        """

        with h5py.File(self.store, mode="r") as h5_handle:
            bins_grp = h5_handle[self.root + "/bins"]
            ann_list = tuple(bins_grp.keys())
        ann_list = [k for k in ann_list if k not in ["chrom", "start", "end"]]
        ann_list.sort()

        return tuple(ann_list)

    def add_bin_annotation(
        self,
        bed_path: str,
        in_file: str = None,
        to_keep: Iterable[str | None] = None,
    ) -> None:
        """Add bin annotation(s) using a bed-like file.

        Add one or more annotation columns to the bins group. One can add:
        - 0/1 column representing an overlap of the bin in the bed-like file
        - any number of annotation columns from the bed-like file

        Parameters
        ----------
        bed_path : str
            Path to the bed-like file to use for the annotation
            ``(chrom, start, end, annot1, ..., annotN)``.
        in_file : str, optional
            Name of the 0/1 annotation column, containing 1 if the bin has at
            least one overlap with any interval in the bed-file, 0 otherwise.
            If None, no such column is created. (default is None)
        to_keep : Iterable[str | None], optional
            Names for the columns of the bed-like file to add to the bins
            group. Names are assigned from left to right (ignoring ``chrom``,
            ``start``, ``end``), and any column that receives a name is kept.
            Any column without a name is discarded. To skip a column, place a
            None in its position. Excess names are ignored. (default is None)

        Notes
        -----
        Adding annotation can drastically increase file size, especially for
        non-numerical annotations; limit string-like non categorical
        annotations (names, ids, ...).
        """

        # TODO: add example to docstring

        def _intersect_dataframes(df_a, df_b):
            """Return dataframe intersection using bedtools intersect -loj"""

            # TODO: Check handling of overlapping annotations in the same file
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
        if in_file in self.list_annotations():
            raise ValueError("'in file' annotation name already exists.")
        if to_keep:
            if set(to_keep) & set(self.list_annotations()):
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
        bins table. Non-existent annotations or default columns (``chrom``,
        ``start``, ``end``) are skipped without raising warning/errors.
        Removed columns still take space, after this process you might want to
        repack the file (see h5repack tool).

        Parameters
        ----------
        to_del : str or Iterable[str]
            String or iterable of them representing bin annotations to remove.
        """

        to_del = self._valid_bin_annotations(to_del)
        with h5py.File(self.store, mode="r+") as h5_handle:
            bin_grp = h5_handle[self.root + "/bins"]
            delete(bin_grp, to_del)

    def annotation_to_ohe(
        self,
        to_ohe: Iterable[str],
        remove_original: bool = False,
        remove_nan_mod: bool = True,
        force_annotation: bool = False,
    ) -> None:
        """Convert bin annotation column(s) to one hot encoding form.

        Given a list of bin annotations, create for each of those columns N
        other columns (where N is the number of modalities, or unique values,
        for that column) each of which in one hot encoding form (1 if modality
        matches, 0 otherwise). If specified, remove the original column.

        Parameters
        ----------
        to_ohe : Iterable[str]
            Iterable of annotations names to perform one-hot encoding on.
            Generated columns are named using ``{original name}_{modality}``.
        remove_original : bool, optional
            Whether to remove the original annotation columns on which ohe is
            performed on. (default is False)
        remove_nan_mod : bool, optional
            Whether to remove columns originated from ohe of the NaN modality,
            meaning ``{original name}_NaN``, if any. (default is True)
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
    ) -> None:
        """Create a cool/mcool file containing only sparsified pixels.

        Generate a new cooler by using as pixels the specified chromosome
        tables filtered according to some alpha values. All bins from the
        file are retained, regardless of whether the corresponding chromosome
        table is present or not.

        Parameters
        ----------
        cool_uri : str
            Where to generate the new cooler. If the specified file does not
            exist it will be created.
        chr_tables : :py:class:`ChromTablesIterator`
            Iterator of pixel tables to merge and use as pixels.
        alpha_thr : str, float or Iterable[float]
            Alpha values to use to filter the pixel tables. If string, compute
            alphas using the specified method (only "optimal" currently). If
            float, use that value as threshold for all tables. If iterable,
            use those values in order, one per table (lengths must match).
        """
        # TODO: Add alpha lenght check
        # TODO: Check the same chromosome was not given twice

        def _filter_alpha_tables(tables, alphas):
            """Filter iterable of tables according to iterable of alphas."""
            for table, alpha in zip(tables, alphas):
                table = table.filter_alpha(alpha)
                yield table[["bin1_id", "bin2_id", "count"]]

        # Adjust input vector of alphas
        if isinstance(alpha_thr, str):
            if alpha_thr == "optimal":
                alpha_thr = ["optimal"] * len(chr_tables)
            else:
                raise ValueError(f"Unknown filtering parameter: {alpha_thr}")
        elif isinstance(alpha_thr, float):
            alpha_thr = [alpha_thr] * len(chr_tables)

        bare_bins = self.bins()[["chrom", "start", "end"]][:]
        filt_pix = _filter_alpha_tables(chr_tables, alpha_thr)

        create_cooler(cool_uri, bins=bare_bins, pixels=filt_pix)
