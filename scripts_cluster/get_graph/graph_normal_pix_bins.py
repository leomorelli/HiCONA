import h5py
import numpy as np
import networkx as nx
import pandas as pd

def get_graph(
    bins: str,
    pix=str,
):
    bins=pd.read_table(bins) #already annotated nodes with pybedtools
    pix=pd.read_table(pix)   #normal pixels (filtered)
    df=cooler.annotate(pix,bins)
    #edge annotation
    g=nx.from_pandas_edgelist(df,source='bin1_id',target='bin2_id',edge_attr='count')
    #node annotation
    node_list=bins[bins.index.isin(np.sort([x for x in g.nodes]))]
    for anno in node_list.columns:
        for n in node_list.index:
            g.nodes[n][anno] = node_list[anno][n] 
    return g

for frac in ['02','05']:
    for cells in [f'GM_count_local_deg_0001_ratio{frac}.txt',f'IMR90_count_local_deg_0005_ratio{frac}.txt',f'HMEC_count_local_deg_01_ratio{frac}.txt',f'HUVEC_count_local_deg_05_ratio{frac}.txt']:
        cell=cells.split('_')[0]
        g=get_graph(pix=f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/sparse_graphs/local_degree/{cells}',bins=f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/sparse_graphs/weighted_sparsification/bins_annotated_{cell}.bed')
        nx.write_graphml(g,f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/sparse_graphs/local_degree/g_{cell}_local_deg{frac}.xml.gz')
