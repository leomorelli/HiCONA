"""Sparsification

Set of functions to perform edge sparsification starting from a COO formatted 
matrix corresponding to Hi-C experiment results. Different sparsification 
methods are supported. The main intended use is through the automatic pipeline
 for conversion of cool-format files into graph-tool objects. 
"""

# TODO: complete module docstring with functions, imports, errors etc.

from math import floor

import networkx as nx
import numpy as np
from pandas import DataFrame
from scipy import integrate


def round_half_up(number: float, decimals: int = 0):
    """Return half way up rounded decimal number
    
    Auxiliary function to round numbers since python default is not what it is
     commonly expected rounding to be. Half way up rounding means "round to 
    closest value, either up or down, and break ties returning upper value".
    """
    multiplier = 10 ** decimals
    return floor(number*multiplier + 0.5) / multiplier


def disparity_index_confidence(pix_df: DataFrame):
    """Add alpha column to dataframe (computed as per Serrano et al. 2009)

    Add a column to the dataframe containing the alpha values for the edges, 
    meaning the confidence level of a weighted edge given local fluctuations 
    in the network, see Serrano et al. 2009 for in depth explaination. Edge 
    iteration is done using an index based approach which takes into account 
    coo properties to speed up the search.

    WARNING: this function might increase substantially the size of the 
    dataframe (+33% or more); edges might be filtered directly to avoid this 
    but it would mean recomputing all alpha values for each threshold to test.

    WARNING: the edge in a disconnected doublet always gets alpha = 1,
    therefore it will be subsequently filtered. This behaviour might not be 
    the desired one and might be changed.
    """

    # TODO: Try preallocating container for the indexes
    # TODO: Try and simplify/split the function
    # TODO: Address doublet problem?
    # TODO: Add alternative break out condition? (maximum iterations maybe)
    # TODO: Right now creating a copy of the dataframe (rm pix_df?)

    # Initialization step
    step = pix_df.shape[0]                          # Number of edges
    flat_df = pix_df.to_numpy().flatten(order="F")  # Dataframe in list form
    alphas = np.ones(step)                          # Non-significant alphas
    bin_id = flat_df[0]                             # Starting bin id

    # Store both indexes and values to minimize lookup
    bin1_ind = 0         # Only one value since only one starting point
    bin1_id = bin_id     # Current value for bin1  HOW TO USE
    bin2_indexes = []    # List of all starting points in column 2
    bin2_ids = []        # List of current values for bin2
    bin2_min = np.infty  # Mock value that should never be equal at the first iteration

    # Initialize part of the stopping criterion
    bin1_ids_left = bin1_ind != step

    # Iterate until edges are exhausted (knowing max bin id is O(E) complex)
    while bin1_ids_left or bin2_indexes:

        # Containers for bin id specific values
        b_indexes = []
        b_values = []

        # Check for bin id in bin2 column
        if bin_id == bin2_min:

            bin2_min_items = [
                (pos, ind[0])
                for pos, ind in enumerate(zip(bin2_indexes, bin2_ids))
                if ind[1] == bin2_min]

            for bin2_pos, bin2_ind in bin2_min_items[::-1]:

                b_indexes.append(bin2_ind - step)
                b_values.append(flat_df[bin2_ind + step])

                # Increment bin2 index, then decide whether to keep or not
                bin2_ind += 1
                bin1_changes = flat_df[bin2_ind-step] != flat_df[bin2_ind-step-1]
                if bin2_ind == 2*step or bin1_changes:
                    del bin2_indexes[bin2_pos], bin2_ids[bin2_pos]
                else:
                    bin2_indexes[bin2_pos] += 1
                    bin2_ids[bin2_pos] = flat_df[bin2_ind]

            bin2_min = np.infty
            if bin2_ids:
                bin2_min = min(bin2_ids)

        # Check for bin in bin1 column
        if bin1_ids_left:
            old_bin1_ind = bin1_ind
            while bin1_id == bin_id:
                bin1_ind += 1
                bin1_id = flat_df[bin1_ind]
                if bin1_ind == step:
                    bin1_ids_left = False
                    break

            # If there was a change in bin1 index, apply changes
            if old_bin1_ind != bin1_ind:
                b_indexes.extend(range(old_bin1_ind, bin1_ind))
                b_values.extend(flat_df[old_bin1_ind+2*step:bin1_ind+2*step])
                bin2_indexes.append(old_bin1_ind + step)
                new_bin2_val = flat_df[old_bin1_ind + step]
                bin2_ids.append(new_bin2_val)
                bin2_min = min(bin2_min, new_bin2_val)

        bin_id += 1

        # Compute alpha values and update
        tot = sum(b_values)  # Sum of weights of neighbouring edges
        k = len(b_values)    # Number of neighbouring edges

        # Normalization does not work on terminal nodes
        if k == 1:
            continue

        new_alphas = [
            1-(k-1)*integrate.quad(lambda x: (1-x)**(k-2), 0, val/tot)[0]
            for ind, val in zip(b_indexes, b_values)]

        for pos, ind in enumerate(b_indexes):
            alphas[ind] = min(alphas[ind], round_half_up(new_alphas[pos], 4))

    # Add alpha values and return dataframe
    flat_df = np.hstack([flat_df, alphas])
    pix_df = DataFrame(
        flat_df.reshape((step,4), order="F"),
        columns=["bin1_id", "bin2_id", "count", "alpha"])
    pix_df = pix_df.astype({"bin1_id": int, "bin2_id": int, "count":int})

    return pix_df


def disparity_network_confidence(pix_df: DataFrame):
    """Add alpha column to dataframe (computed as per Serrano et al. 2009)

    Add a column to the dataframe containing the alpha values for the edges, 
    meaning the confidence level of a weighted edge given local fluctuations 
    in the network, see Serrano et al. 2009 for in depth explaination. A graph
     is built starting from the list of edges, then the procedure is iterated
    over the nodes; the resulting network is converted back to edge list and 
    sorted, since graph traversal does not have inherent order.

    WARNING: this function might increase substantially the size of the 
    dataframe (+33% or more); edges might be filtered directly to avoid this 
    but it would mean recomputing all alpha values for each threshold to test.

    WARNING: the edge in a disconnected doublet always gets alpha = 1,
    therefore it will be subsequently filtered. This behaviour might not be 
    the desired one and might be changed.
    """

    # Add default alpha value
    pix_df["alpha"] = 1

    graph = nx.from_pandas_edgelist(
        pix_df, source='bin1_id', target='bin2_id', edge_attr=True)

    for node in graph:
        k = len(graph[node])

        if k == 1:  # Skip if node only has one neighbour
            continue

        weigths_sum = sum(graph[node][n]["count"] for n in graph[node])

        for neigh in graph[node]:
            norm_weight = graph[node][neigh]["count"] / weigths_sum
            new_alpha = 1 - (k-1)*integrate.quad(
                lambda x, nn=k-2: (1-x)**(nn), 0, norm_weight)[0]

            graph[node][neigh]["alpha"] = min(
                graph[node][neigh]["alpha"], round_half_up(new_alpha, 4))

    pix_df = nx.to_pandas_edgelist(graph, source='bin1_id', target='bin2_id')

    # Requires sorting since graph traversal is not sorted
    # Sort items in pair
    pix_df["bin1_id"], pix_df["bin2_id"] = np.where(
        pix_df["bin1_id"] < pix_df["bin2_id"],
        (pix_df["bin1_id"], pix_df["bin2_id"]),
        (pix_df["bin2_id"], pix_df["bin1_id"]))
    # Sort items in index order
    pix_df.sort_values(["bin1_id", "bin2_id"], inplace=True)

    return pix_df


def add_sparsity_alpha(pix_df: DataFrame, modality: str = "disparity-index"):
    """Add column with sparsification significance values to the dataframe

    Switch case function to define which function should be used to compute 
    the significance level for network sparsification. Column is directly 
    added to the input object. 

    Consider using sparsification.sparsify_network when testing for a single 
    significance cutoff.

    Parameters
    ----------
    pix_df : pandas.DataFrame 
        Dataframe of edges in coo format (bin1, bin2, count)
    modality : str
        Method used to compute alpha values (default: "disparity-index")

    Returns
    -------
    pandas.DataFrame
        Dataframe of pixels in coo format with associated alpha value, hence 
        the rows are in the form (bin1, bin2, count, alpha)
    """

    # TODO: Add better way to show available modality

    if modality == "disparity-index":
        pix_df = disparity_index_confidence(pix_df)
    elif modality == "disparity-network":
        pix_df = disparity_network_confidence(pix_df)
    else:
        raise ValueError(f"{modality} is not a supported modality.")

    return pix_df


def filter_network(
        pix_df: DataFrame,
        cutoff: float = 0.05,
        alpha_col_name: str = "alpha"):
    """Return a dataframe with edges filtered according to significance

    Retrieve only edges with a value less than, or equal to, the specificied 
    cutoff in the specified column, then removes said column. Return a copy, 
    this way multiple cutoff filters can be tested without recomputing the 
    significance levels each time.

    Consider using sparsification.sparsify_network when testing for a single 
    significance cutoff.

    Parameters
    ----------
    pix_df : pandas.DataFrame 
        Dataframe of edges in coo format and realtive alpha value, or other 
        metric to filter on; rows are in the form (bin1, bin2, count, alpha)
    cutoff : float 
        Maximum accepted alpha value (default: 0.05)
    pval_col_name : str
        Column to use when filtering (default: "alpha")

    Returns
    -------
    pandas.DataFrame
        Filtered pixel dataframe
    """

    # TODO: Directly work on the dataframe?

    mask = pix_df[alpha_col_name] <= cutoff
    filtered_df = pix_df[mask].drop(alpha_col_name, axis=1)

    return filtered_df


def sparsify_network(
        pix_df: DataFrame,
        modality: str = "disparity-index",
        cutoff: float = 0.05):
    """Return a dataframe of edges corresponding to the sparsified network
    
    Compute significance level of the weighted edges of a network using some 
    modality, then filter them according to some cutoff.

    Wrapper for simpler use of the workflow sparsification.add_sparsity_alpha 
    + sparsification.filter_network when only one value of alpha is required. 
    In order to test different values of alpha, loop filter_network directly, 
    so to have only one sparsified network in memory at a time (or more if 
    needed, though not suggested with very big networks).

    Parameters
    ----------
    pix_df : pandas.DataFrame 
        Dataframe of edges in coo form (bin1, bin2, count)
    modality : str
        How to compute alpha values (default: "disparity-index")
    cutoff : float 
        Maximum accepted alpha value (default: 0.05)

    Returns
    -------
    pandas.DataFrame
        Sparsified pixels DataFrame
    """

    # TODO: move alpha value to annotations?
    # TODO: add info to cool?

    pix_df = add_sparsity_alpha(pix_df, modality)
    pix_df = filter_network(pix_df, cutoff)

    return pix_df
