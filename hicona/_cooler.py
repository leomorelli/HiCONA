"""Main I/O handling object extending Cooler functionality and format.

Main object to handle .cool/.mcool files in order to perform all the
pre-processing required to obtain chromosome level tables.
Chromosome-level tables are stored in the ``/hicona_tables`` group and
can be retrieved to create filtered networks to analyze.

"""

import json
from typing import Generator, Iterable

import cooler
import h5py
import pandas as pd

from hicona._core import base_table, sparsification, uris
from hicona._ops import bed, hdf5
from hicona.preprocess import Flow
from hicona._table import HiconaTable


__all__ = ["HiconaCooler"]


_TABLES_ROOT = "hicona_tables"
_TABLE_COLUMNS = {
    "bin1_id": "i8",
    "bin2_id": "i8",
    "count": "i4",
    "norm": "f8",
    "alpha_min": "f8",
    "alpha_max": "f8",
}


class HiconaCooler(cooler.Cooler):
    """Class to handle I/O and annotation of cooler-like files.

    Class extending the :class:`cooler.Cooler` class to allow for the creation
    and retrieval of processed pixel tables. Moreover, it allows to easily add
    and remove annotations to/from the "bins" table.

    Parameters
    ----------
    store : str, h5py.File or h5py.Group
        Path to a cooler file, URI string, or open handle to the root HDF5
        group of a cooler data collection.
    kwargs : optional
        Options to be passed to h5py.File upon every access via methods from
        the cooler.Cooler class. These keyword arguments are not passed when
        accessing the HDF5 file via functions from hicona.
        See class constructor for cooler.Cooler for more details.

    Notes
    -----
    The class should not break any functionality of the cooler.Cooler class,
    though it was not tested. Please, report any issue you might find.

    Examples
    --------
    Creating an HiconaCooler object from a cool file path.

    >>> handle = HiconaCooler("path/to/file.cool")
    >>> handle = HiconaCooler("path/to/file.mcool::resolutions/10000")
    """

    # ////////////////////////////////////////////////////////////////////////
    # ////////////////////////// OBJECT DEFINITION ///////////////////////////
    # /////// Class constructor, setter, getters and similar functions ///////
    # ////////////////////////////////////////////////////////////////////////

    def __init__(self, store: str | h5py.File | h5py.Group, **kwargs):
        # Mask deprecated root parameter from super-class
        super().__init__(store, **kwargs)
        self._uris = uris.Uris(self.store, self.root)
        self._tables_root: str = _TABLES_ROOT

    @property
    def tables_root(self) -> str:
        """Return URI of the group in which processed pixel tables are rooted.

        String representing where to store and look for the processed pixel
        tables. This is a relative path starting from the position specified
        by the ``self.root`` attribute.

        Examples
        --------
        >>> handle = HiconaCooler("path/to/file.cool")
        >>> handle.tables_root
        'hicona_tables'  # default value
        """
        return self._tables_root

    def bare_bins(self) -> pd.DataFrame:
        """Get full bin table without any annotation.

        Return the bins table as a pandas.DataFrame with only the columns
        ``chrom``, ``start``, and ``end``.

        Returns
        -------
        DataFrame :
            DataFrame with only the columns ``chrom``, ``start``, and ``end``.

        Examples
        --------
        >>> handle = HiconaCooler("path/to/file.cool")
        >>> handle.bare_bins()
               chrom     start       end
        0       chr1         0     10000
        1       chr1     10000     20000
        2       chr1     20000     30000
        3       chr1     30000     40000
        4       chr1     40000     50000
        ...      ...       ...       ...
        308832  chrY  57180000  57190000
        308833  chrY  57190000  57200000
        308834  chrY  57200000  57210000
        308835  chrY  57210000  57220000
        308836  chrY  57220000  57227415

        [308837 rows x 3 columns]
        """
        return self.bins()[["chrom", "start", "end"]][:]  # type: ignore

    # ////////////////////////////////////////////////////////////////////////
    # /////////////////////////// PUBLIC TABLE API ///////////////////////////
    # // Functions to create, inspect and retrieve sparsified pixel tables ///
    # ////////////////////////////////////////////////////////////////////////

    def _iterate_tables(self) -> Generator[HiconaTable, None, None]:
        """Iterate all saved tables as `HiconaTable` objects."""

        tables_uris = self._uris.add_path(self._tables_root)
        for tab in hdf5.get_keys(tables_uris):
            yield HiconaTable(tables_uris.add_path(tab))

    def _init_raw_table(
        self,
        serial: str,
        method: Flow,
        chunk_size: int,
    ) -> uris.Uris:
        """Initialize a new raw table with the given parameters."""

        # Get table uris
        table_path = f"{self.tables_root}/table_{str(serial).zfill(6)}"
        table_uris = self._uris.add_path(table_path)

        # Initialize table
        num_pix = self.info["nnz"]
        hdf5.init_table(table_uris, num_pix, _TABLE_COLUMNS)

        # TODO: This is currently slow
        # Copy pixel data to the new table
        for lower in range(0, num_pix, chunk_size):
            upper = min(lower + chunk_size, num_pix)
            chunk = self.pixels()[lower:upper]
            assert isinstance(chunk, pd.DataFrame)  # For type checker
            hdf5.write_chunk(table_uris, chunk, lower, chunk.columns)

        # Set table attributes
        table_attrs = {"process_info": json.dumps(method.as_json())}
        hdf5.set_attrs(table_uris, table_attrs)

        return table_uris

    def create_table(
        self,
        ops_flow: str | Flow = "hicona",
        *,
        chunk_size: int = 10_000_000,
    ) -> HiconaTable:
        """Create a normalized and sparsified pixels table.

        Starting from the full pixel table, create a new pixel table by
        filtering, normalizing and then sparsifying the table (see [1]_).
        The exact steps to be performed are specified via a `Flow` object.

        Parameters
        ----------
        ops_flow : str or Flow, optional
            Flow of operations to use to filter and normalize the table. If a
            string is provided, the default flow for the corresponding method
            is used. Default is `hicona`.
        chunk_size : int, optional
            Number of pixels per chunk when processing the raw table.
            Default is 10,000,000.

        Returns
        -------
        HiconaTable :
            The newly created table as a HiconaTable object.

        See Also
        --------
        preprocess.Flow : Used to specify the operations to perform.

        References
        ----------
        .. [1] Serrano et al., "Extracting the multiscale backbone of complex
           weighted networks", PNAS, 2009.

        Examples
        --------
        Create a new table using the default `hicona` method.

        >>> handle = HiconaCooler("path/to/file.cool")
        >>> table = handle.create_table()
        # Logging of the table creation process
        >>> table.dataframe().head()
           alpha_max  alpha_min  bin1_id  bin2_id  count  norm
        0     0.4219     0.4037        1     2720      1   1.0
        1     0.4219     0.3737        1     5476      1   1.0
        2     0.4219     0.3814        1     5802      1   1.0
        3     0.4219     0.3816        1    18419      1   1.0
        4     0.4219     0.4155        3      890      1   1.0
        """

        # TODO: Add a table alias to simplify the fetching

        # Convert any default string to the corresponding flow
        if isinstance(ops_flow, str):
            ops_flow = Flow.from_default(ops_flow)

        # Initialize the tables root if it does not exist already.
        # TODO: Maybe do not hardcode the serial attribute
        table_root_uris = self._uris.add_path(self.tables_root)
        hdf5.require_group(table_root_uris, {"serial": 0})

        # Check there is no table with all matching keywords
        for table in self._iterate_tables():
            if table.flow == ops_flow:
                raise ValueError("E: Table with the same flow already exists.")

        # Get the next available table path
        serial = hdf5.get_attrs(table_root_uris)["serial"]
        hdf5.set_attrs(table_root_uris, {"serial": serial + 1})

        table_uris = self._init_raw_table(serial, ops_flow, chunk_size)
        processor = sparsification.TableProcessor(base_table.Table(table_uris))
        return processor.create_table()

    def fetch_table(self, ops_flow: str | Flow = "hicona") -> HiconaTable:
        """Retrieve a previously created sparsified pixel table.

        Get a previously created sparsified pixel table as a `HiconaTable`
        object. The list of available tables can be printed using the
        `list_tables` method.

        Parameters
        ----------
        ops_flow : str or Flow, optional
            Flow originally used to create the table, provided as either a
            string or a `Flow` object. If a string is provided, the default
            flow for the corresponding method is used. Default is `hicona`.

        Returns
        -------
        HiconaTable :
            The requested table as a HiconaTable object.

        Examples
        --------
        Retrieve a previously created table using the default `hicona` method.

        >>> handle = HiconaCooler("path/to/file.cool")
        >>> table = handle.fetch_table()
        >>> table.dataframe().head()
              alpha_max  alpha_min  bin1_id  bin2_id  count  norm
        0        0.4219     0.4037        1     2720      1   1.0
        1        0.4219     0.3737        1     5476      1   1.0
        2        0.4219     0.3814        1     5802      1   1.0
        3        0.4219     0.3816        1    18419      1   1.0
        4        0.4219     0.4155        3      890      1   1.0
        """

        # TODO: maybe change the method to fetch the table using alias
        # TODO: Sometimes there is an issue with fetching the table
        if isinstance(ops_flow, str):
            ops_flow = Flow.from_default(ops_flow)

        try:
            tables = [t for t in self._iterate_tables() if t.flow == ops_flow]
        except ValueError as exc:
            raise ValueError("E: No table has been generated yet.") from exc

        if len(tables) > 1:  # NOTE: This should never happen
            raise ValueError("E: Multiple tables with matching parameters found.")
        if not tables:
            raise ValueError("E: No table with matching parameters was found.")

        return tables.pop()

    def list_tables(self) -> None:
        """Print the available pixel tables.

        Print the available tables and the operations used to create them in
        a tabular format.

        Notes
        -----
        Graphical representation is still being worked on.
        """

        # TODO: Add a better printout and remove comment from docstring
        # TODO: Add example to docstring when printout is defined
        # TODO: Breaks if the group does not exists

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
        """Return a list of available bin annotation columns.

        Return a list of all available bin annotation columns (that is, all
        columns in the bins group besides `chrom`, `start`, and `end`)
        sorted alphabetically.

        Returns
        -------
        list[str] :
            List of bin annotation names in alphabetical order.

        Examples
        --------
        >>> handle = HiconaCooler("path/to/file.cool")
        >>> handle.bins()[:].columns
        Index(['chrom', 'start', 'end', 'HMM_ann', 'weight'], dtype='object')
        >>> handle.annotation_list()
        ['HMM_ann', 'weight']
        """

        ann_list = hdf5.get_keys(self._uris.add_path("bins"))
        ann_list = [k for k in ann_list if k not in ["chrom", "start", "end"]]
        ann_list.sort()

        return ann_list

    def add_bin_annotation(
        self,
        bed_path: str,
        *,
        in_file: str | None = None,
        to_keep: str | list[str | None] | None = None,
    ) -> None:
        """Add one (or more) bin annotation(s) from a bed-like file.

        Add one or more annotation columns to the bins group. One can add:
        - 0/1 column representing an overlap of the bin in the bed-like file
        - any number of annotation columns from the bed-like file

        A bed-like file is a tab-separated file with the `chrom`, `start`,
        `end` columns followed by any number of other columns.

        Parameters
        ----------
        bed_path : str
            Path to the bed-like file to use for the annotation.
        in_file : str, optional
            Name of the 0/1 annotation column, containing 1 if the bin has at
            least one overlap with any interval in the bed-file, 0 otherwise.
            If `None`, no such column is created. Default is None.
        to_keep : str or list[str | None], optional
            Names for the columns of the bed-like file to add to the bins.
            Names are assigned from left to right ignoring `chrom`, `start`,
            `end`. Any column that receives a name is kept, while any column
            without a name is discarded. To skip a column, place a `None` in
            its position. Excess names are ignored. Default is None.

        Notes
        -----
        Adding annotation can drastically increase file size, especially for
        non-numerical annotations; limit string-like non categorical
        annotations (names, ids, ...).

        Examples
        --------
        Printing a snippet of the bed using pybedtools.

        >>> import pybedtools
        >>> bed = pybedtools.BedTool("path/file.bed")
        >>> print(bed.head(3))
        chr1       0    200    A  0.6
        chr1    2000   4000    A  0.2
        chr1   20400  30000    B  0.3

        Print the head of the bins table.

        >>> handle = HiconaCooler("path/to/file.cool")
        >>> handle.bins()[:].head(3)
            chrom   start    end
        0   chr1       0   10000
        1   chr1   10000   20000
        2   chr1   20000   30000

        Add a 0/1 column indicating overlap with the bed file.

        >>> handle.add_bin_annotation("path/file.bed", in_file="overlap")
        >>> handle.bins()[:].head(3)
            chrom   start    end  overlap
        0   chr1       0   10000        1
        1   chr1   10000   20000        0
        2   chr1   20000   30000        1

        Add only the second annotation column from the bed file (skip first).

        >>> handle.add_bin_annotation("path/file.bed", to_keep=[None, "pval"])
        >>> handle.bins()[:].head(3)
            chrom   start    end  overlap  pval
        0   chr1       0   10000        1   0.6
        1   chr1   10000   20000        0   NaN
        2   chr1   20000   30000        1   0.3
        """

        # TODO: Check behaviour with multiple intersections

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
        ann_df = bed.bed_to_df(bed_path, to_keep)
        ann_df = bed.intersect_dfs(bin_df, ann_df, drop_none=False, loj=True)

        # Check for overlapping annotations
        if len(ann_df) != len(bin_df):
            raise ValueError("E: Overlapping annotations are not supported.")

        # Create OHE column where 1 = "intersection with annotation"
        if in_file:
            col_vals = [1 if c != -1 else 0 for c in ann_df.iloc[:, 5]]
            ann_df[in_file] = pd.Series(col_vals, dtype=bool)

        # Save new annotation columns
        ann_df = ann_df.drop(labels=[None] + list(bin_df.columns), axis=1)
        hdf5.save_table(self._uris.add_path("bins"), ann_df)

    def del_bin_annotation(self, to_del: str | Iterable[str]) -> None:
        """Remove one (or more) bin annotation columns.

        Given one or more bin annotation names, remove those columns from the
        bins table. Non-existent annotations or default columns (`chrom`,
        `start`, `end`) are skipped without raising warning/errors.

        Parameters
        ----------
        to_del : str or Iterable[str]
            String or iterable of them representing bin annotations to remove.

        Notes
        -----
        Removed columns still take space, due to HDF5 specifications. To
        actually reduce the file size, you might want to repack the file
        (see h5repack tool).

        Examples
        --------
        Print the head of the bins table.

        >>> handle = HiconaCooler("path/to/file.cool")
        >>> handle.bins()[:].head(3)
            chrom   start    end  overlap  pval
        0   chr1       0   10000        1   0.6
        1   chr1   10000   20000        0   NaN
        2   chr1   20000   30000        1   0.3

        Remove the `overlap` column.

        >>> handle.del_bin_annotation("overlap")
        >>> handle.bins()[:].head(3)
            chrom   start    end  pval
        0   chr1       0   10000   0.6
        1   chr1   10000   20000   NaN
        2   chr1   20000   30000   0.3
        """

        to_del = self._valid_bin_annotations(to_del)

        if not any(to_del):
            print("W: No valid annotation to delete was provided.")

        hdf5.del_keys(self._uris.add_path("bins"), to_del)

    def ohe_bin_annotation(
        self,
        to_ohe: str | Iterable[str],
        remove_original: bool = False,
        remove_nan_mod: bool = True,
        max_num_mods: int = 10,
    ) -> None:
        """Convert one (or more) bin annotation(s) to one hot encoding form.

        For each specified bin annotation, perform one-hot encoding, that is,
        create a new column for each modality of the annotation, where the
        column is 1 if the bin has that modality, 0 otherwise.

        Parameters
        ----------
        to_ohe : str or Iterable[str]
            Names of the annotations to perform one-hot encoding on.
            Generated columns are named using ``{original name}_{modality}``.
        remove_original : bool, optional
            Whether to remove the original annotation columns on which ohe is
            performed on. Default is False.
        remove_nan_mod : bool, optional
            Whether to remove columns originated from ohe of the NaN modality,
            meaning ``{original name}_NaN``, if any. Default is True.
        max_num_mods : int, optional
            Do not perform ohe if the number of modalities is greater than
            ``max_num_mods``. Default is 10.

        Notes
        -----
        ``max_num_mods`` is a safety measure to avoid huge file size increase
        due to trying to ohe a column with non categorical values.

        Non-existent annotations are skipped without raising warning/errors.

        Removed columns still take space, due to HDF5 specifications. To
        actually reduce the file size, you might want to repack the file
        (see h5repack tool).

        Examples
        --------
        Print the head of the bins table.

        >>> handle = HiconaCooler("path/to/file.cool")
        >>> handle.bins()[:].head(3)
            chrom   start    end  mod
        0   chr1       0   10000    A
        1   chr1   10000   20000    B
        2   chr1   20000   30000    A

        Perform one-hot encoding on the `mod` column.

        >>> handle.ohe_bin_annotation("mod")
        >>> handle.bins()[:].head(3)
            chrom   start    end  mod  mod_A  mod_B
        0   chr1       0   10000    A      1      0
        1   chr1   10000   20000    B      0      1
        2   chr1   20000   30000    A      1      0
        """

        # Select and retrieve needed annotation columns
        to_ohe = self._valid_bin_annotations(to_ohe)
        ann_df: pd.DataFrame = self.bins()[to_ohe][:]  # type: ignore
        # NOTE: currently suppressing type due to messy overloading in cooler

        # If force, skip modalities number check
        too_many = [c for c in to_ohe if ann_df[c].nunique() > max_num_mods]
        if too_many:
            raise ValueError(
                f"E: The variable(s) {', '.join(too_many)} has/have more "
                f"than the default max number of modalities ({max_num_mods})."
                f"\nThis could lead to a huge file size increase. To "
                f"proceed anyway, rerun with force_annotation=True."
            )

        # Generate ohe df and save to hdf5
        ohe_df = pd.get_dummies(ann_df, columns=to_ohe)

        if remove_nan_mod:
            columns = [c for c in ohe_df if not str(c).lower().endswith("_nan")]
            ohe_df = ohe_df[columns]

        hdf5.save_table(self._uris.add_path("bins"), ohe_df)

        # Remove original columns if selected
        if remove_original:
            self.del_bin_annotation(to_ohe)

    def hmm_bin_annotation(
        self,
        ann_file: str,
        *,
        ann_name: str = "hmm",
        nan_annot: str = "Void",
    ) -> None:
        """Add a chromHMM style annotation to the bins table.

        A chromHMM-like annotation is a multimodal annotation without
        overlaps (one region of the genome cannot have two annotations).
        In general, it should also cover the entirety of the genome (though
        this is not strictly required).

        Each bin is annotated with the most enriched annotation, that is, the
        annotation with the highest fold change between the observed and
        expected fraction of bases with that annotation. The expected fraction
        is computed as the fraction of bases with that annotation in the whole
        genome of the bin.

        Parameters
        ----------
        ann_file : str
            Path to the bed file containing the chromHMM-like annotation.
        ann_name : str, optional
            Name of the new annotation column. Default is "hmm".
        nan_annot : str, optional
            Name of the modality replacing gaps. Default is "Void".

        Notes
        -----
        This function was created to add annotations coming from the chromHMM
        software [2]_, though it can be used with any similar annotation.

        References
        ----------

        .. [2] Ernst and Kellis, "Chromatin-state discovery and genome
              annotation with ChromHMM", Nature Protocols, 2017.

        Examples
        --------

        Print the head of the bins table.

        >>> handle = HiconaCooler("path/to/file.cool")
        >>> handle.bins()[:].head(3)
            chrom   start    end
        0   chr1       0   10000
        1   chr1   10000   20000
        2   chr1   20000   30000

        Display the head of the chromHMM-like annotation file.

        >>> import pybedtools
        >>> bed = pybedtools.BedTool("path/file.bed")
        >>> print(bed.head(3))
        chr1   10000  10800   Het
        chr1   10800  13000  Void
        chr1   13000  13200  Prom

        Add the chromHMM-like annotation to the bins table.

        >>> handle.hmm_bin_annotation("path/file.bed")
        >>> handle.bins()[:].head(3)
            chrom   start    end   hmm
        0   chr1       0   10000  Void
        1   chr1   10000   20000   Het
        2   chr1   20000   30000  Prom
        """

        def get_chrom_bed(cool: cooler.Cooler) -> pd.DataFrame:
            """Generate a dataframe in bed-like style for the chromosomes."""

            chrom_info = {
                "chrom": cool.chromnames,
                "start": [0] * len(cool.chromnames),
                "end": cool.chromsizes.values,
            }

            return pd.DataFrame(chrom_info)

        # Compute annotation fractions for both background and query
        annot_col, frac_col = f"{ann_name}_annot", f"{ann_name}_frac"

        ann_table = bed.bed_to_df(ann_file, [annot_col])
        bin_table = self.bare_bins()
        bkg_table = get_chrom_bed(self)

        col_names = annot_col, frac_col
        bin_table = bed.ann_fraction(bin_table, ann_table, col_names, nan_annot)
        bkg_table = bed.ann_fraction(bkg_table, ann_table, col_names, nan_annot)

        out_table = bed.ann_enriched(bin_table, bkg_table, col_names)
        out_table.rename(columns={annot_col: ann_name})

        hdf5.save_table(self._uris.add_path("bins"), out_table)
