"""Placeholder
Placeholder
"""

from cooler import Cooler
from pandas import DataFrame

from sparsification import sparsify_network


def get_chrom_starts(chrom_df: DataFrame, bin_size: int):
    """Return tuple with the indexes of the first bin for each chromosome."""

    chrom_df["num_bins"] = -(-chrom_df["length"] // bin_size)
    chrom_starts = chrom_df["num_bins"].cumsum().tolist()
    chrom_starts = (0, *chrom_starts[:-1])

    return chrom_starts


def get_sequential_bins(
        start: int,
        end: int,
        weight: int,
	    breaks: tuple = tuple()):
    """Return pandas dataframe of edges among genomically contiguous bins.

    It is assumed that bins with sequential ids are genomically contiguous,
    unless specified in the breaks object (usually tuple of indexes of the
    first bin for each chromosome). An arbitrary, homogeneous, weight is 
    assigned to all the edges in COO format in the dataframe. 
	"""

    num_bins = end - start
    indexes = [start + i for i in range(num_bins + 1)]
    bins = [
        (indexes[i], indexes[i+1], weight)
        for i in range(num_bins)
        if indexes[i+1] not in breaks
        ]
    df_bins = DataFrame(bins, columns=("bin1_id", "bin2_id", "count"))

    return df_bins


def bin_threshold(first_bin: int, cutoffs: list):
    """Private auxiliary function to compute inter-chromosome indexer"""
    return min(filter(lambda i: i > first_bin, cutoffs))


def filter_pixels(
        pix_df: DataFrame,
        bin_size: int,
        break_points: list,
        min_count: int = 1,
        max_dist: int = 2e6):
    """Return pandas dataframe containing filtered pixels 

    The applied filters are:
    - below maximal genomic distance of the bins
    - above minimum number of interaction counts
    - non-self looping bins
    - non-inter-chromosomal interactions
    """

    max_diff = -(-max_dist // bin_size)
    pix_df["cutoffs"] = pix_df["bin1_id"].apply(bin_threshold, cutoffs=break_points)

    dist_index = pix_df["bin2_id"] - pix_df["bin1_id"] < max_diff
    min_index = pix_df["count"] > min_count
    self_index = pix_df["bin1_id"] != pix_df["bin2_id"]
    inter_index = pix_df["count"] < pix_df.cutoffs

    filtered_df = pix_df[dist_index & min_index & self_index & inter_index]
    filtered_df.drop("cutoffs", inplace=True, axis=1)

    return filtered_df


def normalize_pixels(
        pix_df: DataFrame,
        bin_df: DataFrame,
        weights_column_name: str,
    ):
    """Return pixels dataframe normalized according to bin weights

    Apply normalization to the pixels by multiplying each by the weights of
    both the corresponding bins.
    """

    # TODO: Also add division normalization
    try:
        weights = bin_df[weights_column_name].tolist()
        pix_df["count"] = pix_df.apply(
            lambda i: weights[i["bin1_id"]]*weights[i["bin2_id"]]*i["count"],
            axis=1)

    except KeyError:
        print(f"WARNING: {weights_column_name} not found in bins dataframe, "
              f"normalization step skipped")

    return pix_df


def annotate_network():
    """Placeholder
    Placeholder
    """
    pass


def cool_to_network(
	    cool_path: str,
	    do_filter_pixels: bool = True,
        do_normalize_pixels: bool = True,
        weights_colum_name: str = "weights",
        do_sparsify_network: bool = True,
        sparsification_method: str = "disparity-network",
        sparsification_cutoff: float = 0.05,
        ):
    """Placeholder
    Placeholder
    """

    # TODO: Allow for handle rather than path
    # TODO: Add path validation
    # TODO: Maybe remove from memory cooler object if not needed 
    # TODO: Add input check
    # TODO: Use dict as option input maybe
    # TODO: Linked option

    cool_handle = Cooler(cool_path)
    pixels = cool_handle.pixels()[:]
    # pixels.to_csv("raw_pixels.csv", index=False)
    bins = cool_handle.bins()[:]
    bin_size = cool_handle.info["bin-size"]
    chrom_starts = get_chrom_starts(cool_handle.chroms()[:], bin_size)
    
    print(f"Num_bins: {bins.shape[0]}")
    print(f"Num_pixels: {pixels.shape[0]}")
    first_bin = pixels.iloc[0]["bin1_id"]
    print(f"First_bin_id: {first_bin}")

    if do_filter_pixels:
        pixels = filter_pixels(pixels, bin_size, chrom_starts)

    pixels.to_csv("filtered_pixels.csv", index=False)

    if do_normalize_pixels:
        pixels = normalize_pixels(pixels, bins, weights_colum_name)

    if do_sparsify_network:
        pixels = sparsify_network(pixels, sparsification_method, sparsification_cutoff)

    # Build net and annotate
    # Decide for optional column annotation (standalone function)
    # Each node: bin index, chrom start end?, 

    return pixels


if __name__ == "__main__":

    import time

    TEST_FILE = "../test_HUVEC.cool"
    # modalities = ["disparity-network"]
    modalities = ["disparity-index"]
    # modalities = ["disparity-network", "disparity-index"]

    for mod in modalities:
        print(f"Now testing modality {mod}")
        start = time.time()
        res = cool_to_network(TEST_FILE, sparsification_method=mod, sparsification_cutoff=2)
        res.to_csv(mod + "_040.csv", index=False)
        end = time.time()
        print(f"It took {end-start}s")
 