"""Placeholder
Placeholder
"""

import networkx as nx
import numpy as np
from pandas import DataFrame
from scipy import integrate


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
    """

    # TODO: Try preallocating container for the indexes 
    # TODO: Add statistics of removed edges (and nodes?)

    # Initialize alpha value to an arbitrarily big number
    step = pix_df.shape[0]
    flat_df = pix_df.to_numpy().flatten(order="F")
    alphas = np.ones(step)
    bin_id = flat_df[0]

    # Indexes of the two columns for iterations
    bin1_ind = 0         # Only one value since only one starting point
    bin2_indexes = []    # List of all starting points in column 2

    bin1_ids_left = bin1_ind != step

    while True:

        # Termination if edges are exhausted
        if (bin1_ind == step) & (not bin2_indexes):
            break
        
        # Termination if maximum limit of iterations is reached
        # if bin1_ind > st

        # Initialization step
        b_indexes = []
        b_values = []

        # Check for bin in column bin2
        bin2_to_remove = []

        for bin2_pos, bin2_ind in enumerate(bin2_indexes):

            if flat_df[bin2_ind] == bin_id:
                b_indexes.append(bin2_ind - step)
                b_values.append(flat_df[bin2_ind + step])

                # Increment bin2 index, then decide whether to keep or not
                bin2_ind += 1

                bin1_changes = flat_df[bin2_ind-step] != flat_df[bin2_ind-step-1]
                if bin2_ind == 2*step or bin1_changes:
                    bin2_to_remove.append(bin2_pos)
                else:
                    bin2_indexes[bin2_pos] += 1

        for bin2_start in reversed(bin2_to_remove):
            del bin2_indexes[bin2_start]

        # Check for bin in column bin1
        if bin1_ids_left:
            old_bin1_ind = bin1_ind
            # print(f"{flat_df[bin1_ind]} == {bin_id}")
            while flat_df[bin1_ind] == bin_id:
                # print("Yes")
                bin1_ind += 1
                if bin1_ind == step:
                    bin1_ids_left = False
                    break

            if old_bin1_ind != bin1_ind:
                # print(f"From old {old_bin1_ind} to new {bin1_ind}")
                b_indexes.extend(range(old_bin1_ind, bin1_ind))
                b_values.extend(flat_df[old_bin1_ind+2*step: bin1_ind+2*step])
                bin2_indexes.append(old_bin1_ind + step)

        # Perform operations on the slice
        tot = sum(b_values)
        k = len(b_values)
        
        # What if k = 1?
        if k > 1:

            new_alphas = [
                1-(k-1)*integrate.quad(lambda x: (1-x)**(k-2), 0, val/tot)[0]
                for ind, val in zip(b_indexes, b_values)
            ]

            for pos, ind in enumerate(b_indexes):
                alphas[ind] = min(alphas[ind], float('%.4f' % new_alphas[pos])) # TODO Change rounding network

        bin_id += 1

    flat_df = np.hstack([flat_df, alphas])
    pix_df = DataFrame(
        flat_df.reshape((step,4), order="F"),
        columns=["bin1_id", "bin2_id", "count", "alpha"])
    pix_df = pix_df.astype({"bin1_id": int, "bin2_id": int, "count":int})

    return pix_df


def disparity_network_confidence(pix_df: DataFrame, weight="count"):
    """Placeholder
    Placeholder
    """

    # TODO: DOES NOT WORK WITH DOUBLETS 
    G = nx.from_pandas_edgelist(pix_df, source='bin1_id', target='bin2_id', edge_attr='count')
    B = nx.Graph()
    for u in G:  # NOTE: This is NOT sorted
        k = len(G[u])
        if k > 1:
            sum_w = sum(np.absolute(G[u][v][weight]) for v in G[u])
            for v in G[u]:
                w = G[u][v][weight]
                p_ij = float(np.absolute(w))/sum_w
                alpha_ij = 1 - (k-1) * integrate.quad(lambda x: (1-x)**(k-2), 0, p_ij)[0]

                if B.has_edge(u,v):
                    if alpha_ij >= B.get_edge_data(u,v)["alpha"]:
                        continue

                B.add_edge(u, v, weight = w, alpha=float('%.4f' % alpha_ij))
        else:  # THIS ACCIDENTALLY OVERRIDES
            for v in G[u]:  # It is just one element
                if not B.has_edge(u, v):
                    B.add_edge(u, v, weight = G[u][v][weight], alpha=1)

    pix_df = nx.to_pandas_edgelist(B)
    pix_df.columns = ["bin1_id", "bin2_id", "count", "alpha"]

    pix_df["bin1_id"], pix_df["bin2_id"] = np.where(
        pix_df["bin1_id"] < pix_df["bin2_id"],
        (pix_df["bin1_id"], pix_df["bin2_id"]),
        (pix_df["bin2_id"], pix_df["bin1_id"])
        )

    pix_df.sort_values(["bin1_id", "bin2_id"], inplace=True)

    return pix_df


def add_sparsity_alpha(pix_df: DataFrame, modality: str = "disparity-index"):
    """Add column with sparsification significance values to the dataframe

    Switch case function to define which function should be used to compute 
    the significance level for network sparsification. Column is directly 
    added to the input object. 

    Consider using sparsification.sparsify_network when testing for a single 
    significance cutoff.

    Keyword arguments:
    pix_df -- (DataFrame) Dataframe of edges in coo form (bin1, bin2, count)
    modality -- (str) How to compute alpha values (default: "disparity-index")
    """

    # TODO Add better way to show available modality

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
    """Return copy of dataframe with edges filtered according to significance

    Retrieve only edges with a value less than, or equal to, the specificied 
    cutoff in the specified column, then removes said column. Return a copy, 
    this way multiple cutoff filters can be tested without recomputing the 
    significance levels each time.

    Consider using sparsification.sparsify_network when testing for a single 
    significance cutoff.

    Keyword arguments:
    pix_df -- (DataFrame) Pandas dataframe with some 
    cutoff -- (float) Maximum accepted alpha value (default: 0.05)
    pval_col_name -- (str) Column to use when filtering (default: "alpha")
    """

    # TODO: Add option to keep alpha values in the annotation?
    mask = pix_df[alpha_col_name] <= cutoff
    # pix_df[mask].drop(alpha_col_name, axis=1)
    return pix_df[mask]


def sparsify_network(
        pix_df: DataFrame,
        modality: str = "disparity-index",
        cutoff: float = 0.05):
    """Return dataframe of edges corresponding to the sparsified network
    
    Compute significance level of the weighted edges of a network using some 
    modality, then filter them according to some cutoff.

    Wrapper for simpler use of the workflow sparsification.add_sparsity_alpha 
    + sparsification.filter_network when only one value of alpha is required. 
    In order to test different values of alpha, loop filter_network directly, 
    so to have only one sparsified network in memory at a time (or more if 
    needed, though not suggested with very big networks).

    Keyword arguments:
    pix_df -- (DataFrame) Dataframe of edges in coo form (bin1, bin2, count)
    modality -- (str) How to compute alpha values (default: "disparity-index")
    cutoff -- (float) Maximum accepted alpha value (default: 0.05)
    """

    # TODO: move alpha value to annotations?
    # TODO: add info to cool?

    pix_df = add_sparsity_alpha(pix_df, modality)
    pix_df = filter_network(pix_df, cutoff)

    return pix_df
