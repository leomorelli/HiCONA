"""Placeholder
Placeholder
"""

from cooler import Cooler
from pandas import DataFrame
from scipy import integrate


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
    pix_df["cutoffs"] = pix_df["bin1_id"].apply(bin_threshold)

    dist_index = pix_df["bin2_id"] - pix_df["bin1_id"] < max_diff
    min_index = pix_df["count"] > min_count
    self_index = pix_df["bin1_id"] != pix_df["bin2_id"]
    inter_index = pix_df["count"] < pix_df.cutoffs

    filtered_df = pix_df[dist_index & min_index & self_index & inter_index]

    return filtered_df


def normalize_pixels(
        pix_df: DataFrame,
        bin_df: DataFrame,
        weights_colum_name: str,
    ):
    """Placeholder
    Placeholder
    """

    # TODO: Add also division normalization
    weights = bin_df[weights_colum_name].tolist()
    pix_df["count"] = pix_df.apply(
        lambda i: weights[i["bin1_id"]]*weights[i["bin2_id"]]*i["count"],
        axis=1)

    return pix_df


def compute_alpha(row, num_neigh, norm_factor):
    """Placeholder
    Placeholder
    """
    new_alpha = 1-(num_neigh-1)*integrate.quad(
        lambda x: (1-x)**(num_neigh-2),
        0, row["count"]/norm_factor
        )[0]
    return min(row["alpha"], new_alpha)


def find_bin_indices():
    """Placeholder
    Placeholder
    """
    pass


def sparsify_network(
        pix_df: DataFrame,
        num_bins: int,
    ):
    """Placeholder
    Placeholder
    !! NOTE it is modifying the original object to avoid creating a new one 
    """

    # TODO: Preallocate??
    # TODO: Add statistics

    num_edges = pix_df.shape[0]
    pix_df["alpha"] = 1  # Arbitrary big so you keep only most significant

    # indexes of the two columns for iterations
    bin1_ind = 0        # Only one value since only one starting point
    bin2_indexes = []   # List of all starting points in column 2

    # There are still indexes in bin1 to iterate over
    bin1_ids_left = True

    for bin_id in range(num_bins):  # For each bin id

        b_indexes = []       # List of indexes of edges touching that bin
        # !!! b_indexes is sorted since all ind in second column are ...
        # ... always smaller than those in the second column
        bin2_to_remove = []  # Starting points of bin2 to remove

        # Check for bin in column bin2
        for bin2_pos, bin2_ind in enumerate(bin2_indexes):  # For each starting point

            curr_row = pix_df.iloc[bin2_ind]

            if curr_row["bin2_id"] == bin_id:
                b_indexes.append(bin2_ind)

                bin2_ind += 1

                if bin2_ind == num_edges:
                    bin2_to_remove.append(bin2_pos)
                elif pix_df.iloc[bin2_ind]["bin1_id"] != curr_row["bin1_id"]:
                    bin2_to_remove.append(bin2_pos)
                else:
                    bin2_indexes[bin2_pos] = bin2_ind

        for bin2_start in reversed(bin2_to_remove):
            del bin2_indexes[bin2_start]

        old_bin1_ind = bin1_ind

        # Check for bin in column bin1
        if bin1_ids_left:
            while (pix_df.iloc[bin1_ind]["bin1_id"] == bin_id) & bin1_ids_left:
                bin1_ind += 1
                if bin1_ind == num_edges:
                    bin1_ids_left = False
                    break

        b_indexes.extend(range(old_bin1_ind, bin1_ind))

        if old_bin1_ind != bin1_ind:
            bin2_indexes.append(old_bin1_ind)

        # Perform operations on the slice
        bin_id_slice = pix_df.iloc[b_indexes]
        norm_factor = bin_id_slice["count"].sum()
        num_neigh = bin_id_slice.shape[0]

        pix_df.loc[b_indexes, "alpha"] = bin_id_slice.apply(
            compute_alpha, num_neigh=num_neigh, norm_factor=norm_factor, axis=1)

        # TODO Add early termination???

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
        weights_colum_name: str = "weights", # TODO: Double check if capitalized
        do_sparsify_network: bool = True,
        sparsification_method: str = "disparity",
        do_keep_annotations: bool = True
        ):
    """Placeholder
    Placeholder
    """

    # TODO: Allow for handle rather than path
    # TODO: Add path validation
    # TODO: Maybe remove from memory cooler object if not needed 
    # TODO: Add input check

    # cool_handle = Cooler(cool_path)
    # pixels = cool_handle.pixels()[:]
    # bins = cool_handle.bins()[:]
    # bin_size = cool_handle.info["bin-size"]
    # chrom_starts = get_chrom_starts(cool_handle.chroms()[:], bin_size)
    
    bin_size = 10000
    pixels = 10000

    if do_filter_pixels:
        pixels = filter_pixels(pixels, bin_size)
 
    if do_normalize_pixels:  # TODO: or later than sparsification?
        pixels = normalize_pixels(pixels, bins, weights_colum_name)
    
    # Sparsification or decay
    # Careful indexes do not match
    if do_sparsify_network:
        pixels = sparsify_network(pixels, 10000)

    # Build net and annotate
    # Decide for optional column annotation (standalone function)
    # Each node: bin index, chrom start end?, 

    return pixels


if __name__=="__main__":
  
    import pandas as pd
    import time

    TEST_PATH_BIG = "../synthetic_files/synt_10000bins_50p_10mu_74611rng.csv"
    TEST_PATH_SMALL = "../synthetic_files/synt_100bins_100p_10mu_81037rng.csv"
    
    start = time.time()
    test_coo = pd.read_csv(TEST_PATH_BIG)
    sparsified = sparsify_network(test_coo, 100)
    print(sparsified)
    stop = time.time()
    print(f"Took {stop-start}")
    
# 87s 25e6 edges, 10000 50 200
