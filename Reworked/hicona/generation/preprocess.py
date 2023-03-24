"""Set of functions to preprocess cools and generate simpler ones

Provides functions for filtering, normalizing and sparsifying the pixels of a 
cool file, then save the results to a new one for further analyses. Only 
preprocess_cool is meant to be directly accessed as of now.
"""

# TODO Fix docs and add proper private hinting

from os.path import exists

from cooler import Cooler, create_cooler
from cooler.fileops import is_cooler
from pandas import DataFrame

from sparsification import sparsify_network


def get_chrom_starts(chrom_df: DataFrame, bin_size: int):
    """Return tuple with the indexes of the first bin for each chromosome.

    WARNING: it is assumed that the bins are sorted the same way as the 
    chromosomes, that is, if the chromosomes are listed as [chrom1, chrom3, 
    chrom2] in the chromosome table, then in the bin table we first find all 
    bin pertaining chromosome 1, then those for chromosome 3 and so on.
    """

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
    
    Create a list of edges in the form (binX, bin(X+1), weight), for each 
    value of X in the range [start, end); weight is an arbitrary, homogeneous 
    weight for all the edges.

    WARNING It is assumed that bins with sequential ids are genomically 
    contiguous, unless specified in the breaks object (usually tuple of 
    indexes of the first bin for each chromosome).  
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


def link_graph(
        pix_df: DataFrame,
        start: int,
        end: int,
        weight: int,
        breaks: tuple = tuple()):
    """ Placeholder
    Placeholder
    """

    # TODO: Right now it works but it is not great
    new_pixels = get_sequential_bins(start, end, weight, breaks)
    pix_df.concat(new_pixels, copy=False)
    pix_df.sort_values(["bin1_id", "bin2_id"], inplace=True)


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
    dist_index = pix_df["bin2_id"] - pix_df["bin1_id"] < max_diff

    chrom1 = sum(pix_df["bin1_id"] > i for i in break_points)
    chrom2 = sum(pix_df["bin2_id"] > j for j in break_points)
    inter_index = chrom1 == chrom2

    min_index = pix_df["count"] > min_count

    self_index = pix_df["bin1_id"] != pix_df["bin2_id"]

    filter_df = pix_df.loc[dist_index & min_index & self_index & inter_index]
    return filter_df


def normalize_pixels(
        pix_df: DataFrame,
        bin_df: DataFrame,
        weights_column_name: str = "weight",
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


def preprocess_cool(
	    cool_in_path: str,
        cool_out_path: str,
	    do_filter_pixels: bool = True,
        do_normalize_pixels: bool = True,
        weights_column_name: str = "weights",
        do_sparsify_pixels: bool = True,
        sparsification_method: str = "disparity-index",
        sparsification_cutoff: float = 0.05,
        ):
    """Create new cool after full preprocessing pipeline

    Given the path of an input cool file, perform different preprocessing 
    options based on the selected flags. Current options are pixel 
    filtering, pixel normalization and pixel sparsification (which are applied
     in this order). 

    Parameters
    ----------
    cool_in_path : str
        Path of the cool file to process in string format
    cool_out_path : str
        Path of the cool file to save the processed data to in string format
    do_filter_pixels: bool, optional
        Apply (or not) filtering to the pixels (default is True)
    do_normalize_pixels: bool, optional
        Apply (or not) normalization to the pixels (default is True)
    weight_column_name: str, optional
        Name of the column in the bins dataframe containing the weights to use
         for normalization. If not found, skip the normalization step (default
         is "weight").
    do_sparsify_pixels: bool, optional
        Apply (or not) sparsification to the pixels (default is True)
    sparsification_method: str, optional
        Method to use for sparsification of the pixels (default is "disparity-
        index").
    
    Returns
    -------
    None
    """

    # TODO: Allow for handle rather than path
    # TODO: Use dict as option input maybe
    # TODO: change info of output file
    # TODO: Better documentation
    # TODO: Better options to select filtering parameters

    # Input validation
    if not isinstance(cool_in_path, str) or not cool_in_path:
        raise ValueError("Path to input cool must be a non empty string")
    if not is_cooler(cool_in_path):
        raise ValueError("Provided cool_in_path is not a valid cool file")
    if not isinstance(cool_out_path, str) or not cool_in_path:
        raise ValueError("Path to input cool must be a non empty string")
    if exists(cool_out_path):
        raise ValueError("Using given output path would overwrite some file")
    if not isinstance(do_filter_pixels, bool):
        raise ValueError("do_filter_pixels option must be a bool")
    if not isinstance(do_normalize_pixels, bool):
        raise ValueError("do_normalize_pixels option must be a bool")
    if not isinstance(weights_column_name, str) or not weights_column_name:
        raise ValueError("Weights column name must be a string")
    if not isinstance(do_sparsify_pixels, bool):
        raise ValueError("do_sparsify_pixels option must be a bool")
    if not isinstance(sparsification_method, str) or not weights_column_name:
        raise ValueError("Sparsification method must be a string")
    if not isinstance(sparsification_cutoff, float):
        raise ValueError("Sparsification cutoff must be a float")

    # Retrierve information from the cool
    cool_handle = Cooler(cool_in_path)
    pixels = cool_handle.pixels()[:]
    bins = cool_handle.bins()[:]
    bin_size = cool_handle.info["bin-size"]
    chrom_starts = get_chrom_starts(cool_handle.chroms()[:], bin_size)
    del cool_handle  # Free memory space

    # Perform preprocessing
    if do_filter_pixels:
        pixels = filter_pixels(pixels, bin_size, chrom_starts)

    if do_normalize_pixels:
        pixels = normalize_pixels(pixels, bins, weights_column_name)

    if do_sparsify_pixels:
        pixels = sparsify_network(
            pixels, sparsification_method, sparsification_cutoff)

    # Generate and save to output cool
    create_cooler(cool_out_path, bins, pixels, ordered=True)


# For testing purposes
if __name__ == "__main__":

    import time

    TEST_FILE_IN = "../test_files/test_HUVEC.cool"
    TEST_FILE_OUT = "../test_files/test_HUVEC_preprocessing_{}.cool"
    modalities = ("disparity-network", "disparity-index")

    for mod in modalities:
        print(f"Now testing modality {mod}")
        start_time = time.time()
        preprocess_cool(TEST_FILE_IN, TEST_FILE_OUT.format(mod))
        end_time = time.time()
        print(f"It took {end_time - start_time}s")
        print("-"*40)
 