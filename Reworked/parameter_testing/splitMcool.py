"""Placeholder
Placeholder
"""

from collections.abc import Iterable
from re import search

import cooler


def split_mcool(
    mcool_path: str,
    file_alias: str,
    output_folder: str,
    filt_thresholds: Iterable[int] = (0, 1, 5),
    max_gen_dist: int = 2e6,
    chrom_patt: str = "^chr",
    verbose: bool = True,
) -> None:
    """Placeholer
    Placeholder
    """

    if not cooler.fileops.is_multires_file(mcool_path):
        raise ValueError("Provided path is not a multi-resolution cooler")
    if verbose:
        print(f"Starting to process {mcool_path}")

    name_template = "file{}_res{}_chrom{}_filt{}"  # Pattern for the output file

    # To allow for sequencial filtering
    filt_thresholds = list(filt_thresholds)
    filt_thresholds.sort()

    for cool_uri in cooler.fileops.list_coolers(mcool_path):
        cool_file = cooler.Cooler(mcool_path + "::" + cool_uri)

        resolution = int(cool_file.info["bin-size"])
        if verbose:
            print(f"\tResolution: {resolution}")

        for chrom in cool_file.chroms()[:].name:
            if search(chrom_patt, chrom) is None:
                print(f"\t\t{chrom} does not match chromosome pattern, skipping")
                continue

            if verbose:
                print(f"\t\tChromosome: {chrom}")

            chrom_extrema = cool_file.extent(chrom)
            ch_pix = cool_file.pixels().fetch(chrom)
            ch_pix["distance"] = (ch_pix["bin2_id"] - ch_pix["bin1_id"]) * resolution
            indexer = ch_pix[
                (ch_pix["distance"] > max_gen_dist)
                | (ch_pix["bin2_id"] < chrom_extrema[0])
                | (ch_pix["bin2_id"] > chrom_extrema[1])
                | (ch_pix["bin1_id"] == ch_pix["bin2_id"])
            ].index
            ch_pix.drop(indexer, inplace=True)
            ch_pix.drop(["bin1_id", "bin2_id"], axis=1, inplace=True)

            for thr in filt_thresholds:
                if verbose:
                    print(f"\t\t\tThreshold: {thr}")

                file_name = name_template.format(file_alias, resolution, chrom, thr)
                ch_pix[ch_pix["count"] > thr].to_csv(
                    output_folder + file_name, index=False
                )


if __name__ == "__main__":
    MCOOL_PATH = "../test_files/Rao_2014_HUVEC_MboI_4DNFIRMZ7QTE.mcool"
    FILE_ALIAS = "Rao_2014_HUVEC"
    OUT_FOLDER = "../test_files/split_mcools/"
    split_mcool(MCOOL_PATH, FILE_ALIAS, OUT_FOLDER)
