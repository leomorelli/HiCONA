"""Main I/O handling object extending Cooler functionality and format.

Main object to handle .cool/.mcool files in order to perform all the
pre-processing required to obtain chromosome level tables.
Chromosome-level tables are stored in the ``/hicona_tables`` group and
can be retrieved to create filtered networks to analyze.
"""

from collections.abc import Iterable
from typing import Generator
import json

import cooler
import h5py
import pandas as pd

from .hicona_table import RawTable, HiconaTable, _TablesIterator
from .settings import HICONA_SETTINGS
from .processing.processing_flow import ProcessingFlow
from .processing.table_processor import TableProcessor
from .uris import Uris
from .utils.bed_ops import ann_enriched, ann_fraction, bed_to_df, intersect_dfs
from .utils.hdf5_ops import (
    require_group,
    del_keys,
    get_attrs,
    get_keys,
    save_table,
    set_attrs,
    init_table,
    write_chunk,
)


__all__ = ["HiconaCooler"]


class HiconaCooler(cooler.Cooler):
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
    # ////////////////////////// OBJECT DEFINITION ///////////////////////////
    # /////// Class constructor, setter, getters and similar functions ///////
    # ////////////////////////////////////////////////////////////////////////

    # TODO: autodetect system resources to define better chunk size

    def __init__(self, store: str | h5py.File | h5py.Group, **kwargs):
        # Mask deprecated root parameter from super-class
        super().__init__(store, **kwargs)
        self._chunk_size = HICONA_SETTINGS.parameters.base_pix_chunk
        self._tables_root = "hicona_tables"  # TODO: maybe move to configs

    @property
    def chunk_size(self):
        """Size of fixed length chunks used during processing."""
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, value):
        min_val = HICONA_SETTINGS.parameters.min_pix_chunk
        if not (isinstance(value, int)) or value < min_val:
            raise ValueError(f"chunk_size must be: int >= {min_val}.")
        self._chunk_size = value

    @property
    def tables_root(self):
        """Return partial uri of the pixel tables starting from root."""
        return self._tables_root

    def bare_bins(self) -> pd.DataFrame:
        """Get full bin table without any annotation."""
        return self.bins()[["chrom", "start", "end"]][:]  # type: ignore

    # ////////////////////////////////////////////////////////////////////////
    # /////////////////////////// PUBLIC TABLE API ///////////////////////////
    # // Functions to create, inspect and retrieve sparsified pixel tables ///
    # ////////////////////////////////////////////////////////////////////////

    def _iterate_tables(self) -> Generator[HiconaTable, None, None]:
        """Iterate all saved tables as `HiconaTable` objects."""

        for tab in get_keys(self.store, f"{self.root}/{self._tables_root}"):

            # Get table uris
            table_path = f"{self._tables_root}/{tab}"
            table_uris = Uris(self.store, self.root, table_path)

            # TODO: check whether the table is valid and skip if not
            # Get table processing parameters
            table_attrs = get_attrs(*table_uris.hdf5_uris())
            flow_json = json.loads(table_attrs["process_info"])
            params = ProcessingFlow.from_json(flow_json)

            yield HiconaTable(table_uris, params, self.binsize, self._chunk_size)

    def _init_raw_table(self, serial: str, method: ProcessingFlow) -> Uris:
        """Initialize a new raw table with the given parameters."""

        # Get table uris
        table_path = f"{self.tables_root}/table_{str(serial).zfill(6)}"
        table_uris = Uris(self.store, self.root, table_path)

        # Initialize table
        cols = HICONA_SETTINGS.conventions.table_columns
        num_pix = self.info["nnz"]
        init_table(*table_uris.hdf5_uris(), num_pix, cols)

        # Copy pixel data to the new table
        chunk_size = self._chunk_size
        for lower in range(0, num_pix, chunk_size):
            upper = min(lower + chunk_size, num_pix)
            chunk = self.pixels()[lower:upper]
            write_chunk(*table_uris.hdf5_uris(), chunk, lower, chunk.columns)

        # Set table attributes
        table_attrs = {"process_info": json.dumps(method.as_json())}
        set_attrs(*table_uris.hdf5_uris(), table_attrs)

        return table_uris

    def create_table(self, flow: str | ProcessingFlow = "hicona"):
        """Create a normalized and sparsified version of the pixels table.

        Create a new table using the specified normalization procedure,
        then add the sparsification scores to all pixels of said table.
        The filters and normalization methods can be either provided via
        a ProcessingFlow object or as a string, in which case the default
        flow for the corresponding method is used.

        Parameters
        ----------
        flow : str or ProcessingFlow, optional
            Flow to use to filter and normalize the table. If a string is
            provided, the default flow for the corresponding method is used.
            (default is "hicona")
        """

        # Convert any default string to the corresponding flow
        if isinstance(flow, str):
            flow = ProcessingFlow.from_default(flow)

        # Initialize the tables root if it does not exist already.
        # TODO: Maybe do not hardcode the serial attribute
        table_root_uris = Uris(self.store, self.root, self.tables_root)
        require_group(*table_root_uris.hdf5_uris(), {"serial": 0})

        # Check there is no table with all matching keywords
        for table in self._iterate_tables():
            if table.flow == flow:
                raise ValueError("E: Table with the same flow already exists.")

        # Get the next available table path
        serial = get_attrs(*table_root_uris.hdf5_uris())["serial"]
        set_attrs(*table_root_uris.hdf5_uris(), {"serial": serial + 1})

        table_uris = self._init_raw_table(serial, flow)
        table = RawTable(table_uris, flow, self.binsize, self._chunk_size)
        processor = TableProcessor(table)
        processor.create_table()

    def fetch_table(self, flow: str | ProcessingFlow = "hicona") -> HiconaTable:
        """Retrieve an previously created `HiconaTable` object.

        Retrieve a previously created table using the specified method.
        The flow can be either provided via a ProcessingFlow object or as
        string, in which case the default scheduler for the corresponding
        method is used.

        Parameters
        ----------
        flow : str or ProcessingFlow
            Method used to filter and normalize the table. If a string is
            provided, the default scheduler for the corresponding method is used.
            (default is "hicona").

        Returns
        -------
        HiconaTable :
            The requested table as a HiconaTable object.
        """

        if isinstance(flow, str):
            flow = ProcessingFlow.from_default(flow)

        try:
            tables = [t for t in self._iterate_tables() if t.flow == flow]
        except ValueError as exc:
            raise ValueError("E: No table has been generated yet.") from exc

        if len(tables) > 1:  # NOTE: This should never happen
            raise ValueError("E: Multiple tables with matching parameters found.")
        if not tables:
            raise ValueError("E: No table with matching parameters was found.")

        return tables.pop()

    def list_tables(self) -> None:
        """Print available chromosome tables for each set of parameters."""
        # TODO: Add a better printout

        out = ""
        separator = "-" * 78 + "\n"

        for table in self._iterate_tables():
            out += separator
            for k, v in table.flow.as_json().items():
                out += f"- {k}: {v}\n"

        if not out:
            out = "No tables available yet.\n"

        out = separator + out + separator
        print(out.strip())

    # ////////////////////////////////////////////////////////////////////////
    # ///////////////////////// ANNOTATION FUNCTIONS /////////////////////////
    # ////////////////// Add/process bin annotation columns //////////////////
    # ////////////////////////////////////////////////////////////////////////

    def _valid_bin_annotations(self, names: str | Iterable[str]) -> list[str]:
        """Return only valid bin annotation names as iterable of strings"""

        names = names or []
        names = [names] if isinstance(names, str) else names
        names = [n for n in names if n in self.annotation_list()]

        return names

    def annotation_list(self) -> list[str]:
        """Return an iterable of available bin annotation columns.

        Return an iterable of all available bin annotation columns (that is,
        all columns in the bins group besides ``chrom``, ``start``, and
        ``end``) sorted alphabetically.

        Returns
        -------
        Iterable[str] :
            Iterable of bin annotation names in alphabetical order.
        """

        ann_list = get_keys(self.store, self.root + "/bins")
        ann_list = [k for k in ann_list if k not in ["chrom", "start", "end"]]
        ann_list.sort()

        return ann_list

    def add_bin_annotation(
        self,
        bed_path: str,
        in_file: str | None = None,
        to_keep: str | list[str | None] | None = None,
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
        to_keep : str | None | list[str | None], optional
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
        to_keep = [to_keep] if isinstance(to_keep, str) else to_keep
        # to_keep = to_keep if any(to_keep) and to_keep else None

        # Check for no overlap in old and new annotations
        if in_file in self.annotation_list():
            raise ValueError("E: 'in file' annotation name already exists.")
        if to_keep and any(ann in to_keep for ann in self.annotation_list()):
            raise ValueError("E: Overlap with old annotations, stopping.")

        # Create the two bin df and merge on default bed columns
        # While reading, replace chrom, start, end of bed file with None.
        bin_df = self.bare_bins()
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
        save_table(self.store, "/".join([self.root, "bins"]), ann_df)

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

        if not any(to_del):
            print("W: No valid annotation to delete was provided.")

        del_keys(self.store, self.root + "/bins", to_del)

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
        ann_df: pd.DataFrame = self.bins()[to_ohe][:]  # type: ignore
        # NOTE: currently suppressing type due to messy overloading in cooler

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
            columns = [c for c in ohe_df if not str(c).lower().endswith("_nan")]
            ohe_df = ohe_df[columns]

        save_table(self.store, "/".join([self.root, "bins"]), ohe_df)

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

        def get_chrom_bed(cool):
            """Generate a dataframe in bed-like style for the chromosomes."""

            chrom_info = {
                "chrom": cool.chromnames,
                "start": [0] * len(cool.chromnames),
                "end": cool.chromsizes.values,
            }

            return pd.DataFrame(chrom_info)

        # Compute annotation fractions for both background and query
        annot_col, frac_col = f"{ann_name}_annot", f"{ann_name}_frac"

        ann_table = bed_to_df(ann_file, [annot_col])
        bin_table = self.bare_bins()
        bkg_table = get_chrom_bed(self)

        col_names = annot_col, frac_col
        bin_table = ann_fraction(bin_table, ann_table, col_names, nan_annot)
        bkg_table = ann_fraction(bkg_table, ann_table, col_names, nan_annot)

        out_table = ann_enriched(bin_table, bkg_table, col_names)
        out_table.rename(columns={annot_col: ann_name})

        save_table(self.store, "/".join([self.root, "bins"]), out_table)

    # ////////////////////////////////////////////////////////////////////////
    # /////////////////////// MISCELLANEOUS FUNCTIONS ////////////////////////
    # ////////// Any function not falling in the previous categories /////////
    # ////////////////////////////////////////////////////////////////////////

    def gen_sparsified_cooler(
        self,
        cool_uri: str,
        chr_tables: _TablesIterator,
        alpha_thr: str | float | Iterable[float | str],
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
        chr_tables : :py:class:`_TablesIterator`
            Iterator of pixel tables to merge and use as pixels.
        alpha_thr : str, float or Iterable[float | str]
            Alpha values to use to filter the pixel tables. If string, compute
            alphas using the specified method (only "optimal" currently). If
            float, use that value as threshold for all tables. If iterable,
            use those values in order, one per table (lengths must match).
        """
        # TODO: Add alpha length check
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
        bare_bins = self.bare_bins()
        cooler.create_cooler(cool_uri, bins=bare_bins, pixels=filt_pix)
