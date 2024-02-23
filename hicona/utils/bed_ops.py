"""Operations on bed-like files (e.i. any operation using bedtools)."""

import numpy as np
from pybedtools import BedTool


def bed_to_df(bed, non_def_names: str | list[str] | None = None):
    """Placeholder"""

    if isinstance(bed, str):
        bed = BedTool(bed)

    if isinstance(non_def_names, str):
        non_def_names = [non_def_names]

    colnames = ["chrom", "start", "end"]
    if non_def_names:
        colnames += non_def_names

    # Pad with None columns
    num_columns = bed.field_count()
    if len(colnames) < num_columns:
        colnames += [None] * (num_columns - len(colnames))

    dataf = bed.to_dataframe(
        disable_auto_names=True,
        comment="#",
        header=None,
    )
    dataf.columns = colnames  # Set after to allow duplicate names

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

    # Suppress linting error due to pybedtools wrapper implementation
    # pylint: disable=unexpected-keyword-arg, too-many-function-args
    bed_a = BedTool.from_dataframe(df_a)
    bed_b = BedTool.from_dataframe(df_b)
    bed_a = bed_a.intersect(bed_b, **kwargs)
    # pylint: enable=unexpected-keyword-arg, too-many-function-args

    inters = bed_to_df(bed_a, col_names[3:])
    if drop_none:
        inters.drop(labels=[None], axis=1, inplace=True)

    return inters


def ann_fraction(query, ref, colnames, nan_annot=None):
    """Get fraction of bases with given annotation in an interval."""

    anno_col, frac_col = colnames
    inters = intersect_dfs(query, ref, frac_col, wao=True)

    # Compute fraction of bases with annotation in the interval
    bin_size = inters["end"] - inters["start"]
    inters[frac_col] = inters[frac_col] / bin_size

    # NOTE: HMM annotation should cover the chromosomes entirely,
    # though currently there is a variable sized gap (usually 10000
    # bp) at the beginning of almost all chromosomes. Currently fixing
    # manually the gap by assigning some annoatation value.
    # TODO: Fix issue above
    if nan_annot:
        indexer = inters[anno_col].isna()
        inters.loc[indexer, anno_col] = nan_annot
        inters.loc[indexer, frac_col] = 1

    # Sum the fractions for two identical annotations in the interval
    grouping_cols = [c for c in inters.columns if c != frac_col]
    inters = inters.groupby(grouping_cols, as_index=False).sum()
    # NOTE: Each bin fraction should be 1, but that is not the case
    # Is it cause chromHMM is for non-coding regions?

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
