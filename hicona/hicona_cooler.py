"""Main I/O handling object extending Cooler functionality and format.

Main object to handle .cool/.mcool files in order to perform all the
pre-processing required to obtain chromosome level tables.
Chromosome-level tables are stored in the ``/hicona_tables`` group and
can be retrieved to create filtered networks to analyze.
"""

from collections.abc import Iterable
import re
import time

from cooler import Cooler, create_cooler
import h5py
import pandas as pd

from .hicona_table import HiconaTable, HiconaTablesIterator
from .settings import HICONA_SETTINGS
from .table_processor import TableProcessor
from .utils.bedops import ann_enriched, ann_fraction, bed_to_df, intersect_dfs
from .utils.decorators import wait_hdf5_lock
from .utils.misc import parse_regions
from .utils.tableops import get_dataf_mapping

__all__ = ["HiconaCooler"]


class HiconaCooler(Cooler):
    """An extension of the Cooler class to prepare data for network analysis.

    :py:class:`HiconaCooler` inherits from :py:class:`cooler.Cooler` and
    extends it by adding new functionalities, mainly revolving around
    the creation of chromosome-level tables to use for network analyses.
    Tables are stored in a separate group (``hicona_tables``) of the
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
        self._chunk_size = HICONA_SETTINGS.parameters.base_pix_chunk

    @property
    def chunk_size(self):
        """Size of fixed lenght chunks used during processing."""
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, value):
        min_val = HICONA_SETTINGS.parameters.min_pix_chunk
        if not (isinstance(value, int)) or value < min_val:
            raise ValueError(f"chunk_size must be: int >= {min_val}.")
        self._chunk_size = value

    @wait_hdf5_lock
    def extent(self, region: str) -> tuple[int]:
        """Return bin IDs of the lower and upper bounds of a genomic region.

        Wrapper of the extend method from the parent Cooler class in order to
        be able to implement parallelization. See Cooler class documentation.
        """
        return super().extent(region)

    # ////////////////////////////////////////////////////////////////////////
    # //////////////////////////// I/O FUNCTIONS /////////////////////////////
    # ////////// Functions to create or retrieve tables and groups ///////////
    # ////////////////////////////////////////////////////////////////////////

    @wait_hdf5_lock
    def _init_table(self, grp_path, tab_size, col_mapping):
        """Initialize dataframe columns as 1D arrays."""

        with h5py.File(self.store, mode="r+") as h5_handle:
            grp = h5_handle.require_group(grp_path)
            for name, dtype in col_mapping.items():
                grp.require_dataset(
                    name,
                    shape=(tab_size,),
                    dtype=dtype,
                    compression="gzip",
                )

    def _place_table(self, grp_path, chunks, cols=None):
        """Place pixel chunk in table at the given position."""

        @wait_hdf5_lock
        def put(store, grp_path, names, chunk, lower):
            with h5py.File(store, mode="r+") as h5_handle:
                grp = h5_handle[grp_path]
                for name in names:
                    grp[name][lower : lower + len(chunk)] = chunk[name]

        # Make single pandas df into interable of chunks
        chunks = [chunks] if isinstance(chunks, pd.DataFrame) else chunks

        lower = 0
        for chunk in chunks:
            names = cols if cols else chunk.columns
            put(self.store, grp_path, names, chunk, lower)
            lower += len(chunk)

    def _save_table(self, grp_path, table):
        """Convenience shorthand to initialize and place table at once."""

        self._init_table(grp_path, len(table), get_dataf_mapping(table))
        self._place_table(grp_path, table)

    def _bare_bins(self):
        """Get full bin table without any annotation."""
        return self.bins()[["chrom", "start", "end"]][:]

    def _get_chrom_bed(self):
        """Generate a dataframe in bed-like style for the chromosomes."""

        bed = pd.DataFrame(
            {
                "chrom": self.chromnames,
                "start": [0] * len(self.chromnames),
                "end": self.chromsizes.values,
            }
        )

        return bed

    # ////////////////////////////////////////////////////////////////////////
    # /////////////////// PRIVATE PRE-PROCESSING FUNCTIONS ///////////////////
    # // Functions to pass from full-pixel table to chromosome-level tables //
    # ////////////////////////////////////////////////////////////////////////

    def _create_table(self, region, table_root, filt_opts):
        """Create chromosome-level table given the set of parameters."""

        @wait_hdf5_lock
        def does_not_exist(region, store, table_root):
            """Check whether chromosome was already processed."""

            with h5py.File(store, mode="r") as h5_handle:
                table = h5_handle[table_root]
                answer = region not in table.keys()
            return answer

        @wait_hdf5_lock
        def get_pix_idx(borders, store, table_root):
            """Placeholder."""

            with h5py.File(store, mode="r") as h5_handle:
                h5_grp = h5_handle[table_root]
                return [h5_grp["indexes/bin1_offset"][b] for b in borders]

        def get_queries(binsize, upper_idx, dist_thr, count_thr, quant_thr):
            """Get a dictionary containing the strings to use as queries."""

            queries = {}

            # QUERY: remove pixels with bins outside of the interval
            queries["out_interval"] = f"bin2_id < {upper_idx}"

            # QUERY: remove self-looping pixels
            queries["self_looping"] = "bin1_id != bin2_id"

            # QUERY: remove pixels above maximal genomic distance
            max_diff = -(-dist_thr // binsize)
            queries["genomic_dist"] = f"bin2_id - bin1_id < {max_diff}"

            # QUERY: remove pixels with raw counts below a certain theshold
            if count_thr > 0:
                queries["below_counts"] = f"count > {count_thr}"

            # QUERY: remove a quantile of pixels from the processed table
            if quant_thr > 0:
                queries["quantile_thr"] = quant_thr

            return queries

        if does_not_exist(region, self.store, table_root):
            print(f"Starting to work on {region}")
            bin_idx = self.extent(region)
            pix_idx = get_pix_idx(bin_idx, self.store, self.root)

            table = HiconaTable(self.store, self.root + "/pixels", pix_idx)
            queries = get_queries(self.binsize, bin_idx[1], **filt_opts)
            processor = TableProcessor(table, queries)

            table_path = table_root + "/" + region
            self._init_table(
                table_path,
                processor.table_size,
                HICONA_SETTINGS.conventions.table_columns,
            )
            self._place_table(table_path, processor.get_processed_chunks())

        else:
            print(f"W: {region} already processed with these params, skip.")

    @wait_hdf5_lock
    def _init_tables_grp(self, dist_thr, count_thr, quant_thr):
        """Initialize main table group and param specific group if needed."""

        with h5py.File(self.store, mode="r+") as h5_handle:
            root_template = HICONA_SETTINGS.conventions.table_uri_template
            table_root = root_template.format(dist_thr, count_thr, quant_thr)
            table_root = self.root + "/hicona_tables/" + table_root

            # Create container group if not already existent and set attrs
            table_grp = h5_handle.require_group(table_root)
            table_grp.attrs["count-threshold"] = count_thr
            table_grp.attrs["distance-threshold"] = dist_thr
            table_grp.attrs["quantile-threshold"] = quant_thr

        return table_root

    # ////////////////////////////////////////////////////////////////////////
    # /////////////////////////// PUBLIC TABLE API ///////////////////////////
    # // Functions to create, inspect and retrieve chromosome-level tables ///
    # ////////////////////////////////////////////////////////////////////////

    def create_tables(
        self,
        chrom_selection: str | Iterable[str] = "humanCanonical",
        dist_thr: int = 200_000_000,
        count_thr: int = 0,
        quant_thr: float = 0.05,
    ):
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
            Remove pixels whose raw count is not greater than this value.
            (default is 0)
        quant_thr: float, optional
            Remove pixels whose normalized counts are below this percentile.
            (default is 0.0)
        """

        # Initialize table container
        filt_opts = {
            "dist_thr": dist_thr,
            "count_thr": count_thr,
            "quant_thr": quant_thr,
        }
        table_root = self._init_tables_grp(**filt_opts)

        start = time.time()

        # Create chromosome-level groups and datasets
        for region in parse_regions(chrom_selection, self._get_chrom_bed()):
            self._create_table(region, table_root, filt_opts)

        print(f"Took {time.time()-start}s")

    def list_tables(self) -> None:
        """Print available chromosome tables for each set of parameters."""

        # TODO: Maybe find a prettier, sorted and more flexible way to print
        try:
            with h5py.File(self.store, mode="r") as h5_handle:
                tables_grp = h5_handle[self.root + "/hicona_tables"]
                par_str = "PARAMETER SETS:"
                for par_grp in tables_grp.values():
                    par_str += "\n" + "-" * 78
                    par_lst = [f"\n-{k}: {v}" for k, v in par_grp.attrs.items()]
                    par_str += "".join(par_lst)
                    par_str += "\n-intervals:"
                    par_str += "".join([f"\n\t--{k}" for k in par_grp.keys()])
                par_str += "\n" + "-" * 78
            print(par_str)

        except KeyError:
            print("E: No tables have been created yet.")

    def tables(
        self,
        chrom_selection: str | Iterable[str] = "humanCanonical",
        dist_thr: int = None,
        count_thr: int = None,
        quant_thr: float = None,
    ) -> HiconaTablesIterator:
        """Return an iterator of selected tables and respective information.

        Use the input parameters to define which tables to retrieve, then
        return a :py:class:`HiconaTablesIterator` where each item is a tuple in
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
        dist_thr: int, optional
            Fetch tables created using this value as distance threshold.
            If None, get all tables regardless of the used value.
        count_thr: int, optional
            Fetch tables created using this value as count threshold.
            If None, get all tables regardless of the used value.
        quant_thr: float, optional
            Fetch tables created using this value as quantile threshold.
            If None, get all tables regardless of the used value.

        Returns
        -------
        :py:class:`HiconaTablesIterator`:
        """

        # Create valid groups regex according to input parameters
        chroms = parse_regions(chrom_selection, self._get_chrom_bed())
        dist_thr = dist_thr or r"\d+"
        count_thr = count_thr or r"\d+"
        quant_thr = quant_thr or r"[\d.]+(\.[\d]+)?"
        root_template = HICONA_SETTINGS.conventions.table_uri_template
        grp_template = root_template.format(dist_thr, count_thr, quant_thr)
        grp_regex = re.compile(f"^{grp_template}$")

        # Define a list of partial URIs to valid tables
        with h5py.File(self.store, mode="r") as h5_handle:
            tables_grp = h5_handle[self.root + "/hicona_tables"]
            valid_grps = [g for g in tables_grp if grp_regex.match(g)]
            valid_tables = []
            for grp in valid_grps:
                tabs = [grp + "/" + c for c in chroms if c in tables_grp[grp]]
                valid_tables.extend(tabs)

        valid_tables = [f"{self.root}/hicona_tables/{t}" for t in valid_tables]
        return HiconaTablesIterator(self.store, valid_tables)

    # ////////////////////////////////////////////////////////////////////////
    # ///////////////////////// ANNOTATION FUNCTIONS /////////////////////////
    # ////////////////// Add/process bin annotation columns //////////////////
    # ////////////////////////////////////////////////////////////////////////

    def _valid_bin_annotations(self, names: str | Iterable[str]):
        """Return only valid bin annotation names as iterable of strings"""

        names = names or []
        names = [names] if isinstance(names, str) else names
        names = [n for n in names if n in self.annotation_list()]

        return names

    def annotation_list(self) -> Iterable[str]:
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

        return ann_list

    def add_bin_annotation(
        self,
        bed_path: str,
        in_file: str = None,
        to_keep: str | None | Iterable[str | None] = None,
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
        to_keep : str | None | Iterable[str | None], optional
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

        # Convert to_keep to None if all elements are None
        to_keep = list(to_keep) if isinstance(to_keep, tuple) else to_keep
        to_keep = to_keep if isinstance(to_keep, list) else [to_keep]

        # Check for no overlap in old and new annotations
        if in_file in self.annotation_list():
            raise ValueError("E: 'in file' annotation name already exists.")
        if any(ann in to_keep for ann in self.annotation_list()):
            raise ValueError("E: Overlap with old annotations, stopping.")

        # Create the two bin df and merge on default bed columns
        # While reading, replace chrom, start, end of bed file with None.
        bin_df = self._bare_bins()
        ann_df = bed_to_df(bed_path, to_keep)
        ann_df = intersect_dfs(bin_df, ann_df, drop_none=False, loj=True)

        # Check for overlapping annotations
        if len(ann_df) != len(bin_df):
            raise ValueError("E: Overlapping annotations are not supported.")

        # Create OHE column where 1 = "intersection with annotation"
        if in_file:
            col_vals = [1 if c != -1 else 0 for c in ann_df.iloc[:, 5]]
            ann_df[in_file] = pd.Series(col_vals, dtype=bool)

        # Save new annotation columns
        ann_df = ann_df.drop(labels=[None] + list(bin_df.columns), axis=1)
        self._save_table("bins", ann_df)

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
            for annot in to_del:
                del bin_grp[annot]

        if not any(to_del):
            print("W: No valid annotation to delete was provided.")

    def ohe_bin_annotation(
        self,
        to_ohe: str | Iterable[str],
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
        to_ohe : str | Iterable[str]
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
            max_mods = HICONA_SETTINGS.parameters.max_annot_mods
            too_many = [c for c in to_ohe if ann_df[c].nunique() > max_mods]
            if too_many:
                raise ValueError(
                    f"E: The variable(s) {', '.join(too_many)} has/have more "
                    f"than the default max number of modalities ({max_mods})."
                    f"\nThis could lead to a huge file size increase. To "
                    f"proceed anyway, rerun with force_annotation=True."
                )

        # Generate ohe df and save to hdf5
        ohe_df = pd.get_dummies(ann_df, columns=to_ohe)

        if remove_nan_mod:
            columns = [c for c in ohe_df if not c.lower().endswith("_nan")]
            ohe_df = ohe_df[columns]

        self._save_table("bins", ohe_df)

        # Remove original columns if selected
        if remove_original:
            self.del_bin_annotation(to_ohe)

    def hmm_bin_annotation(
        self,
        ann_file: str,
        ann_name: str = "HMM",
        nan_annot: str = "Void",
    ):
        """Add chromHMM style annotation to the bins table.

        Given a chromHMM-like annotation (multimodal, covering the entire
        genome), add an annotation column to the bins table, where the
        modality is the annotation which is most enriched in the bin with
        respect to the reference chromosome (fold change between observed
        bases with the annotation and expected ones).

        Parameters
        ----------
        ann_file : str
            Path to the bed file containing the chromHMM annotation.
        """

        # Compute annotation fractions for both background and query
        annot_col, frac_col = f"{ann_name}_annot", f"{ann_name}_frac"

        ann_table = bed_to_df(ann_file, annot_col)
        bin_table = self._bare_bins()
        bkg_table = self._get_chrom_bed()

        col_names = annot_col, frac_col
        bin_table = ann_fraction(bin_table, ann_table, col_names, nan_annot)
        bkg_table = ann_fraction(bkg_table, ann_table, col_names, nan_annot)

        out_table = ann_enriched(bin_table, bkg_table, col_names)
        out_table.rename(columns={annot_col: ann_name})

        self._save_table("bins", out_table)

    # ////////////////////////////////////////////////////////////////////////
    # /////////////////////// MISCELLANEOUS FUNCTIONS ////////////////////////
    # ////////// Any function not falling in the previous categories /////////
    # ////////////////////////////////////////////////////////////////////////

    def gen_sparsified_cooler(
        self,
        cool_uri: str,
        chr_tables: HiconaTablesIterator,
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
        chr_tables : :py:class:`HiconaTablesIterator`
            Iterator of pixel tables to merge and use as pixels.
        alpha_thr : str, float or Iterable[float]
            Alpha values to use to filter the pixel tables. If string, compute
            alphas using the specified method (only "optimal" currently). If
            float, use that value as threshold for all tables. If iterable,
            use those values in order, one per table (lengths must match).
        """
        # TODO: Add alpha lenght check
        # TODO: Check the same chromosome was not given twice

        def tables_generator(tables, alphas):
            """Filter iterable of tables according to iterable of alphas."""
            default_bin_cols = ["bin1_id", "bin2_id", "count"]
            for table, alpha in zip(tables, alphas):
                yield table.get_dataframe(alpha)[default_bin_cols]

        # Adjust input vector of alphas
        if isinstance(alpha_thr, str):
            if alpha_thr == "optimal":
                alpha_thr = ["optimal"] * len(chr_tables)
            else:
                raise ValueError(f"E: Unknown filtering param: {alpha_thr}")
        elif isinstance(alpha_thr, float):
            alpha_thr = [alpha_thr] * len(chr_tables)

        filt_pix = tables_generator(chr_tables, alpha_thr)
        create_cooler(cool_uri, bins=self._bare_bins(), pixels=filt_pix)
