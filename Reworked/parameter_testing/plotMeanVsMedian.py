"""Placeholder
Placeholder
"""

from collections.abc import Iterable
from os import listdir
import re

from numpy import log2, log10
import matplotlib.pyplot as plt
from pandas import concat, DataFrame, read_csv
import seaborn as sns

# VALUE IMMEDIATELY BECOMES CUTOFF


def plot_mean_vs_median(
    csv_folder: str,
    chrom_id: str,
    output_folder: str,
    res_range: Iterable[int] = (1e4, 5e4),
    max_distance: int = 2e4,
) -> None:
    """Placeholder
    Placeholders
    """

    # List of csvs regarding the chromosome of interest
    csv_list = [c for c in listdir(csv_folder) if re.search(f"_chrom{chrom_id}_", c)]

    # List of filtering values for that chromosome
    filt_list = [re.search(r"_filt\d+", csv)[0] for csv in csv_list]
    filt_list = list({int(re.search(r"\d+", val)[0]) for val in filt_list})
    filt_list.sort(reverse=True)

    # List of resolutions for that chromosome
    resolution_list = [re.search(r"_res\d+", csv)[0] for csv in csv_list]
    resolution_list = list({int(re.search(r"\d+", val)[0]) for val in resolution_list})
    resolution_list.sort()

    # Plotting aesthetics
    flier_props = {"marker": ".", "markersize": 0.5}

    for filt_val in filt_list:
        filt_val = 1
        # Dataframes to store statistics at different resolutions
        mean_df = DataFrame(columns=("count", "distance", "resolution"))
        median_df = DataFrame(columns=("count", "distance", "resolution"))

        for res in resolution_list:
            # Only consider resolutions within the specified range
            if res < res_range[0] or res > res_range[1]:
                continue

            # Retrieve dataframe
            file_re = re.compile(rf".+_res{res}_chrom{chrom_id}_filt{filt_val}")
            file_name = list(filter(file_re.match, csv_list))[0]
            res_df = read_csv(csv_folder + "/" + file_name)

            # Plot starting count distribution
            sns.boxplot(
                data=res_df, x="distance", y="count", flierprops=flier_props
            ).set_title("test")
            plt.xticks(rotation="vertical")
            plt.yscale("log")
            plt.show()
            plt.clf()

            norm_data = res_df
            norm_data["mean_norm"] = norm_data.groupby("distance")["count"].transform(
                "median"
            )
            norm_data["mean_norm"] = log2(
                norm_data["count"] / norm_data["mean_norm"] + 1
            )
            sns.set(rc={"figure.figsize": (15, 8)})
            sns.boxplot(
                data=norm_data, x="distance", y="mean_norm", flierprops=flier_props
            )
            plt.xticks(rotation="vertical")
            plt.show()
            plt.clf()

            expected_mean = res_df.groupby("distance").mean().reset_index()
            expected_mean["resolution"] = res
            mean_df = concat([mean_df, expected_mean])

        mean_plot = sns.lineplot(mean_df, x="distance", y="count", hue="resolution")
        out_path = output_folder + "/" + f"mean_chr{chrom_id}_filt{filt_val}"
        mean_plot.get_figure().savefig(out_path)
        plt.show()
        plt.clf()


if __name__ == "__main__":
    TEST_PATH = "../test_files/split_mcools"
    plot_mean_vs_median(TEST_PATH, "chr19", "test_graphs")
