"""Operations on bed-like files (e.i. any operation using bedtools)."""

import numpy as np
import pandas as pd
from pybedtools import BedTool


def bed_to_df(
    bed: str | BedTool,
    col_names: list[str | None] | None = None,
) -> pd.DataFrame:
    """Convert bed file to pandas dataframe, with custom column names.

    Convert a bed file to a pandas dataframe, with the following caveats:
    - Replace "." with np.NaN
    - Set the first three column names to "chrom", "start", "end"
    - Set all other columns to the provided names, or None if not provided
    """

    if isinstance(bed, str):
        bed = BedTool(bed)

    colnames = ["chrom", "start", "end"]
    if col_names:
        colnames += col_names

    # Pad with None columns
    num_columns = int(bed.field_count())
    if (empty_fields := num_columns - len(colnames)) > 0:
        colnames += [None] * empty_fields

    dataf = bed.to_dataframe(
        disable_auto_names=True,
        comment="#",
        header=None,
    )
    dataf = pd.DataFrame(dataf) if not isinstance(dataf, pd.DataFrame) else dataf
    dataf.columns = tuple(colnames)  # Set after to allow duplicate names

    # "." is the empty intersection for bedtools loj
    # Set after instead of using na_values="." due to type guessing issue
    dataf.replace(".", np.NaN, inplace=True)

    return dataf


def get_intersection_names(df_a, df_b, inters_names=None, rm_standard=True):
    """Placeholder"""

    colnames = list(df_a.columns)
    if rm_standard:
        colnames += [None] * 3 + list(df_b.columns)[3:]
    else:
        colnames += list(df_b.columns)

    if inters_names:
        if isinstance(inters_names, str):
            inters_names = [inters_names]
        colnames += inters_names

    return colnames


def intersect_dfs(df_a, df_b, inters_names=None, drop_none=True, **kwargs):
    """Return dataframe intersection using bedtools intersect.

    Placeholder
    """

    col_names = get_intersection_names(df_a, df_b, inters_names)

    bed_a = BedTool.from_dataframe(df_a)
    bed_b = BedTool.from_dataframe(df_b)
    bed_a = bed_a.intersect(bed_b, **kwargs)  # pylint: disable=E1121 #type: ignore

    inters = bed_to_df(bed_a, col_names[3:])
    if drop_none:
        inters.drop(labels=[None], axis=1, inplace=True)

    return inters


def ann_fraction(query, ref, colnames, nan_annot="NaN"):
    """Get fraction of bases with given annotation in an interval."""

    anno_col, frac_col = colnames
    inters = intersect_dfs(query, ref, frac_col, wao=True)

    # Compute fraction of bases with annotation in the interval
    bin_size = inters["end"] - inters["start"]
    inters[frac_col] = inters[frac_col] / bin_size

    # NOTE: To my knowledge, chromHMM should cover the entire genome, which
    # does not happen all the time somehow. Hence the following code.

    # If an interval does not have any intersection, it will have a row with
    # NaN annotation and 0 base overlap. Replace overlap fraction to 1.
    inters.loc[inters[anno_col].isna(), frac_col] = 1
    inters[anno_col].replace(np.NaN, nan_annot, inplace=True)

    # If only partial overlap, need to fill the rest with NaN annotation
    # So create a dataset with missing rows and add it to the original
    partial = inters.groupby(["chrom", "start", "end"])["HMM_frac"].sum()
    partial = partial[partial < 1].reset_index()
    partial["HMM_frac"] = 1 - partial["HMM_frac"]
    partial[anno_col] = nan_annot
    inters = pd.concat([inters, partial], ignore_index=True)

    # Sum the fractions for two identical annotations in the interval
    grouping_cols = [c for c in inters.columns if c != frac_col]
    inters = inters.groupby(grouping_cols, as_index=False).sum()

    # Assert that fractions sum to one, keeping in mind floating point errors
    grouped = inters.groupby(["chrom", "start", "end"])[frac_col].sum()
    max_shift = (grouped - 1).abs().max()
    tolerance = 1e-12
    assert max_shift < tolerance

    return inters


def ann_enriched(query, ref, colnames):
    """Get fold change of the annotation over the background."""

    base_cols = ["chrom", "start", "end"]
    anno_col, frac_col = colnames
    anno_bkg, frac_bkg = f"{anno_col}_bkg", f"{frac_col}_bkg"

    ref = ref.rename(columns={anno_col: anno_bkg, frac_col: frac_bkg})
    inters = intersect_dfs(query, ref, loj=True)

    # Keep only the lines where the annotation and bkg match
    inters.query(f"{anno_col} == {anno_bkg}", inplace=True)

    # For each bin keep the annotation with the highest fold change
    inters["fold_change"] = inters[frac_col] / inters[frac_bkg]
    index = inters.groupby(base_cols)["fold_change"].idxmax()
    inters = inters.loc[index][[anno_col]]

    return inters
