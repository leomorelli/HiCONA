"""Placeholder
Placeholder
"""

from os.path import exists

from numpy import nan
from pandas import DataFrame, get_dummies
from pybedtools import BedTool


def annotate_bins(
        bins_df: DataFrame,
        annot_bed: str,
        in_file_annot_name: str = None,
        other_annot_names: list = None,
        annot_to_encode: list = None,
        force_annotation: bool = False):
    """Return bins dataframe annotated using a bed-like file
    
    Given a bins dataframe, add one or more annotation columns corresponding 
    to the presence of the bin in the bed-like file and/or any of the bin 
    annotations present in the bed-like file. Note that this could drastically
     increase file size.

    Parameters
    ----------
    bins_df : pandas.DataFrame
        Dataframe containing the set of bins to annotate. Bins can already be 
        annotated, the new annotation columns will be appendend at the end of 
        the currently existing ones.
    annot_bed : str
        Path of the bed-like file (chrom, start, end, annot1, ..., annotN) to 
        use for the annotation is string format.
    in_file_annot_name : str, optional
        If not None, create an annotation column (using this sting as name) 
        with 1 if the bin has an intersection in the bed-like file, 0 
        otherwise (default is None)
    other_annot_name : list, optional
        List of names for the annotation columns in the bed-like file to keep 
        in the final annotated dataframe. This list must have length less than
         or equal to the number of annotation columns in the bed-like file.
        Names are assigned from left to right, and any annotation who received
        a name is kept. Any column without a name (or to which None was given)
         is automatically discarded. (default is None)
    annot_to_encode : list, optional
        List of names of the annotations to split using one-hot encoding. The 
        original column is dropped and the derived ones are named using 
        {original name}_{modality}. (default is None)
    force_annotation : bool, optional
        Trying to perform one-hot encoding on annotations with many modalities
         will issue and error in order to prevent huge file bloating. To 
        suppress the error and split anyway change to True.(dafault is False)

    Returns
    -------
    Pandas dataframe of annotated bins in bed-like format (chrom, start, end,
    annot1, ..., annotN).
    """

    # TODO: docstrings
    # TODO: General reworking, not it is not great
    # TODO: check new colname is valid
    # TODO: technically should check if non empty string or iterable, not None
    # TODO: check it is a bed file (with file_type?)
    # TODO: check no provided colnames are identical to previous ones
    # TODO: add warning if nans in ohe or too many cathegorical modalities
    # TODO: annotation info?
    # TODO: add a way to encode after the file has been generated

    max_num_modalities = 10  # Critical number of modalities for a variable

    # Check inputs
    if not isinstance(bins_df, DataFrame):
        raise ValueError("Bins must be provided in a pandas DataFrame.")
    if not isinstance(annot_bed, str):
        raise ValueError("Path to annotation bed-like file must be a string.")
    if not exists(annot_bed):
        raise ValueError("Path to annotation file must be valid.")
    if not isinstance(in_file_annot_name, str) and in_file_annot_name:
        raise ValueError("in_file_annot_name must be a string.")
    if not isinstance(other_annot_names, list) and other_annot_names:
        raise ValueError("Column names must be either a list or None.")
    if not isinstance(annot_to_encode, list) and annot_to_encode:
        raise ValueError("Columns to encode must be either a list or None.")
    if not isinstance(force_annotation, bool):
        raise ValueError("force_annotation must be a bool.")

    # Save starting column names and define positions for new column names
    initial_col_names = bins_df.columns.tolist()
    initial_stop = len(initial_col_names)
    if other_annot_names:
        other_start = initial_stop + 3
        other_stop = other_start + len(other_annot_names)

    # Create bedtools objects and merge
    bin_bedtool = BedTool.from_dataframe(bins_df)
    annot_bedtool = BedTool(annot_bed)

    # NOTE: suppressing warning due to pybedtools way of creating the wrapper
    # pylint: disable=unexpected-keyword-arg, too-many-function-args
    bin_bedtool = bin_bedtool.intersect(annot_bedtool, loj=True)
    # pylint: enable=unexpected-keyword-arg, too-many-function-args

    # Convert result to pandas, rename and remove columns
    annotated_bins = bin_bedtool.to_dataframe()
    final_names = [None] * len(annotated_bins.columns)
    final_names[0:initial_stop] = initial_col_names
    if in_file_annot_name:
        final_names[initial_stop+2] = in_file_annot_name
    if other_annot_names:
        if other_stop > len(annotated_bins.columns):
            print("WARNING: Provided number of column annotations exceeds "
                "the number of available columns, skipping this step")
        else:
            final_names[other_start:other_stop] = other_annot_names
    annotated_bins.columns = final_names
    annotated_bins.drop(labels=[None], axis=1, inplace=True)

    if in_file_annot_name:
        annotated_bins[in_file_annot_name] = [
            1 if cell != -1 else 0
            for cell in annotated_bins[in_file_annot_name]]

    # Replace all cells containing only a dot with NaNs
    # "." is default for bedtools loj in non-matching non-positional columns
    annotated_bins.replace(r"^\.$", nan, regex=True, inplace=True)

    if annot_to_encode and other_annot_names:

        # Remove all elements not present in the new column annotations
        annot_to_encode = [
            annot for annot in annot_to_encode
            if annot in annotated_bins.columns]

        if not annot_to_encode:
            print("WARNING: None of the annotation to encode is valid.")

        if not force_annotation:
            too_many_vals = [
                col_name for col_name in annot_to_encode
                if annotated_bins[col_name].nunique() > max_num_modalities]

            if too_many_vals:
                raise ValueError(f"The variable(s) {', '.join(too_many_vals)}"
                    f"has/have more than the maximum number of modalities "
                    f"allowed ({max_num_modalities}). This could lead to "
                    f"massive inflation of the matrix. If you wish to proceed"
                    f" rerun the function with force_annotation=True")

        annotated_bins = get_dummies(
            annotated_bins, prefix=annot_to_encode, columns=annot_to_encode)

    return annotated_bins


def delete_annotations(bins_df: DataFrame, annot_names: list):
    """ Remove annotations from the bins dataframe

    Remove any number of annotations from the bins dataframe by providing a 
    list of annotation names to remove (in place). Skip not-found annotations.

    Parameters
    ----------
    bins_df : pandas.Dataframe
        Dataframe containing annotated bins in bed-like format
    annot_names: list
        List of annotation names to remove from the dataframe
    """

    for name in annot_names:
        if name in bins_df.columns:
            bins_df.drop(name, axis=1, inplace=True)
        else:
            print(f"WARNING: There is no {name} annotation column, skipping.")


def rename_annotation(bins_df: DataFrame, annot_dict: list):
    """ Rename annotations in the bins dataframe

    Rename any number of annotations from the bins dataframe by providing a 
    dictionary for old_name to new_name mapping. Skip not-found annotations.

    Parameters
    ----------
    bins_df : pandas.Dataframe
        Dataframe containing annotated bins in bed-like format
    annot_names: dict
        Dictionary to use when renaming the annotation columns. Keys are the 
        current annotation names, values are the new annotation names.
    """

    bins_df.rename(columns=annot_dict, inplace=True)


if __name__ == "__main__":

    import cooler

    COOL_PATH = "../test_files/test_HUVEC.cool"
    BED_PATH = "../test_files/promoter_with_info.bed"
    # BED_PATH = "../test_files/only_promoter.bed"
    COL_NAMES = ["promoter_ENSG", "strand", "promoter_name"]
    TO_ENCODE = ["strand"]

    c = cooler.Cooler(COOL_PATH)
    bins_dataf = c.bins()[:]

    out = annotate_bins(bins_dataf, BED_PATH, "is_promoter", COL_NAMES, TO_ENCODE)

    print(out[~out.isnull().any(axis=1)])
    print("-"*100)
    print("DIMENSION")
    print(out.shape)
    print("-"*100)
    print("MEMORY USAGE")
    print(out.memory_usage(deep=True))
