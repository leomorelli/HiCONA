"""Placeholder
Placeholder
"""

import networkx as nx
import numpy as np
from pandas import DataFrame
from scipy import integrate




def find_bin_indexes(
        pix_df: DataFrame,
        bin_id: int,
        bin1_ind: int,
        bin2_indexes: list):
    """Return the list of indexes of pixels containing the specified bin
    
    Using an efficient iterative procedure, taking into account the properties
     of the coo format, retrieve the list of indexes of pixels containint the 
    specified bin (and the updated indexes for the iterative procedure). By 
    design of the procedure, the list of indexes is always sorted.
    """

    # TODO: If lookup is too time expensive, consider storing start point
    # TODO: If needed, find alterantive to storing bin2 to remove

    # Initialization step
    b_indexes = []
    num_edges = pix_df.shape[0]

    # Check for bin in column bin2
    bin2_to_remove = []
    for bin2_pos, bin2_ind in enumerate(bin2_indexes):

        curr_row = pix_df.iloc[bin2_ind]

        if curr_row["bin2_id"] == bin_id:
            b_indexes.append(bin2_ind)

            # Increment bin2 index, then decide whether to keep or not
            bin2_ind += 1
            if bin2_ind == num_edges:
                bin2_to_remove.append(bin2_pos)
            elif pix_df.iloc[bin2_ind]["bin1_id"] != curr_row["bin1_id"]:
                bin2_to_remove.append(bin2_pos)
            else:
                bin2_indexes[bin2_pos] = bin2_ind

    for bin2_start in reversed(bin2_to_remove):
        del bin2_indexes[bin2_start]

    # Check for bin in column bin1
    bin1_ids_left = bin1_ind == num_edges
    if bin1_ids_left:
        old_bin1_ind = bin1_ind
        while (pix_df.iloc[bin1_ind]["bin1_id"] == bin_id) & bin1_ids_left:
            bin1_ind += 1
            if bin1_ind == num_edges:
                bin1_ids_left = False
                break

        if old_bin1_ind != bin1_ind:
            b_indexes.extend(range(old_bin1_ind, bin1_ind))
            bin2_indexes.append(old_bin1_ind)

    return b_indexes, bin1_ind, bin2_indexes


def alpha_disparity(row, num_neigh, norm_factor):
    """Return alpha value according to Serrano et al. 2009 (private use)"""
    alpha = 1-(num_neigh-1)*integrate.quad(
        lambda x: (1-x)**(num_neigh-2), 0, row["count"]/norm_factor)[0]
    return min(row["alpha"], alpha)


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
    pix_df["alpha"] = 1

    # Indexes of the two columns for iterations
    bin1_ind = 0         # Only one value since only one starting point
    bin2_indexes = []    # List of all starting points in column 2
    
    # Start from minimum bin id in the pixel matrix
    bin_id = pix_df.iloc[0]["bin1_id"]
    max_id = max(pix_df["bin2_id"]) + 1

    while bin_id < max_id:
        
        # Termination if edges are exhausted
        if (bin1_ind == pix_df.shape[0]) & (not bin2_indexes):
            break

        # Retrieve indexes for that bin
        b_indexes, bin1_ind, bin2_indexes = find_bin_indexes(
            pix_df, bin_id, bin1_ind, bin2_indexes)

        # Perform operations on the slice
        bin_id_slice = pix_df.iloc[b_indexes]
        norm_factor = bin_id_slice["count"].sum()
        num_neigh = bin_id_slice.shape[0]

        pix_df.loc[b_indexes, "alpha"] = bin_id_slice.apply(alpha_disparity,
            num_neigh=num_neigh, norm_factor=norm_factor, axis=1)

        bin_id += 1


def disparity_network_confidence(pix_df: DataFrame, weight="count"):
    """Placeholder
    Placeholder
    """
    
    # TODO: DOES NOT WORK WITH DOUBLETS 
    G = nx.from_pandas_edgelist(pix_df, source='bin1_id', target='bin2_id', edge_attr='count')
    B = nx.Graph()
    for u in G:
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
    pix_df = nx.to_pandas_edgelist(B)
    
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
        disparity_index_confidence(pix_df)
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
    return pix_df[mask].drop(alpha_col_name, axis=1)


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
