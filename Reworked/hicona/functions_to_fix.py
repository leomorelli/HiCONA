from pybedtools import BedTool
from pandas import DataFrame, get_dummies
from numpy import nan


# TODO: fix, since from before being Cooler subclass
def annotate_bins(
    self,
    annot_bed: str,
    in_file_annot_name: str = None,
    other_annot_names: list = None,
) -> None:
    """Annotate bins using a bed-like file.

    Add one or more annotation columns; one can add:
    - 0/1 column representing the presence of the bin in the bed-like file
    - any number of columns from the bed-like file (regardless of type)
    NOTE: annotation can drastically increase matrix size, especially for
    non-numerical annotations; limit categorical annotations (names...).

    Parameters
    ----------
    annot_bed : str
        Path to the bed-like file (chrom, start, end, annot1, ..., annotN)
        to use for the annotation in string format.
    in_file_annot_name : str, optional
        If not None, create a 0/1 annotation column (using this string as
        name) with 1 if the bin has (at least) one intersection in the
        bed-like file, 0 otherwise. (default is None)
    other_annot_names : list, optional
        If not None, use the elements of this list (or iterable) as names
        for the annotation columns in the bed-like file to add to the bins
        dataframe. Names are assigned from left to right (ignoring the
        first 3 columns), and any column that received a name is kept.
        Any column without a name (number of names is less than the number
        of columns in the bed-like - 3), or to which None was given, is
        discarded. Excess names are discarded. (default is None)
    """

    # TODO: check new colname is valid
    # TODO: check it is a bed file (with file_type?)
    # TODO: check no provided colnames are identical to previous ones
    # TODO: annotation info?

    # Save starting column names and define positions for new column names
    initial_col_names = self.bins.columns.tolist()
    initial_num_cols = len(initial_col_names)
    if in_file_annot_name:
        if in_file_annot_name in initial_col_names:
            raise ValueError("'in file' annotation name already exists.")
    if other_annot_names:
        annot_pos_start = initial_num_cols + 3
        annot_pos_stop = annot_pos_start + len(other_annot_names)

    # NOTE: suppressed error due to pybedtools wrapper implementation
    # pylint: disable=unexpected-keyword-arg, too-many-function-args
    bin_bedtool = BedTool.from_dataframe(self.bins)
    annot_bedtool = BedTool(annot_bed)
    bin_bedtool = bin_bedtool.intersect(annot_bedtool, loj=True)
    # pylint: enable=unexpected-keyword-arg, too-many-function-args

    # Convert result to pandas, rename and remove columns
    annotated_bins = bin_bedtool.to_dataframe()
    final_names = [None] * len(annotated_bins.columns)
    final_names[0:initial_num_cols] = initial_col_names
    if in_file_annot_name:
        final_names[initial_num_cols + 2] = in_file_annot_name
    if other_annot_names:
        final_names[annot_pos_start:annot_pos_stop] = other_annot_names
    annotated_bins.columns = final_names
    annotated_bins.drop(labels=[None], axis=1, inplace=True)

    # Convert in_file column to 0/1
    if in_file_annot_name:
        annotated_bins[in_file_annot_name] = [
            1 if cell != -1 else 0 for cell in annotated_bins[in_file_annot_name]
        ]

    # Replace all cells containing only a dot with NaNs
    # "." is default for bedtools loj in non-matching non-positional columns
    annotated_bins.replace(r"^\.$", nan, regex=True, inplace=True)

    self.bins = annotated_bins


# TODO: fix, since from before being Cooler subclass
def encode_annotation(
    self,
    to_encode: list = None,
    force_annotation: bool = False,
) -> None:
    """Convert bin annotation column(s) to one hot encoding form

    Given a list of column names, replace each of those columns with N
    other columns (where N is the number of modalities for that column)
    each of which in one hot encoding form (1 if modality matches, 0
    otherwise). Ignore not-found columns.

    to_encode : list, optional
        List (or iterable) of names of the annotations to split using
        one-hot encoding. The original column is dropped and the derived
        ones are named using {original name}_{modality}. (default is None)
    force_annotation : bool, optional
        Trying to perform one-hot encoding on annotations with many
        modalities will issue and error in order to prevent huge file
        bloating. To suppress the error and split anyway change to True.
        (dafault is False)
    """

    max_num_modalities = 10  # Critical number of allowed modalities

    # Remove all elements not present in the new column annotations
    to_encode = [ann for ann in to_encode if ann in self.bins.columns]

    if not to_encode:
        print("WARNING: No valid annotation to encode.")

    if not force_annotation:
        too_many_vals = [
            col_name
            for col_name in to_encode
            if self.bins[col_name].nunique() > max_num_modalities
        ]

        if too_many_vals:
            raise ValueError(
                f"The variable(s) {', '.join(too_many_vals)}"
                f"has/have more than the maximum number of modalities "
                f"allowed ({max_num_modalities}). This could lead to "
                f"massive inflation of the matrix. If you wish to proceed"
                f" rerun the function with force_annotation=True"
            )

    self.bins = get_dummies(self.bins, columns=to_encode)


# FOR LONG DISTANCE DECAY
# from numpy import exp
# def decay_function(x, a, b, c):
#     return a * exp(-b * x) + c
# decay_curve = group_counts.agg(stat).to_frame()
# decay_curve.reset_index(inplace=True)
# decay_curve.plot(x="bin_difference", y="count")
# plt.show()

# popt, _ = curve_fit(
#     decay_function, decay_curve["bin_difference"], decay_curve["count"]
# )
# plt.plot(
#     decay_curve["bin_difference"],
#     decay_function(decay_curve["bin_difference"], *popt),
# )

# ALTERNATIVE WRAPPER
# def integral_wrapper(*args, **kwargs):
#     int_key = str(args) + str(kwargs)
#     int_val = int_cache.get(int_key)
#     if not int_val:
#         int_val = func(*args, **kwargs)
#         int_cache[int_key] = int_val
#     return int_val
