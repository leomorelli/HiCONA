import networkit as nk
import networkx as nx
import cooler 
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
import time
import seaborn as sns




for cell in cells:
    c=cooler.Cooler(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/mcool/{cell}::resolutions/10000')
    c_id=cell.split('_')[2]
    pix=pd.read_table(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/sparse_graphs/weighted_sparsification/{c_id}_count_filtered_pix.bed')
    bins=c.bins()[:]
    anno=cooler.annotate(pix,bins)
    g_nx=nx.from_pandas_edgelist(anno,source='bin1_id',target='bin2_id')

    lista_nx= []
    for e in g_nx.edges():
        lista_nx.append([e[0], e[1]])
    df_nx=pd.DataFrame(lista_nx)
    lista_nx= []
    for e in g_nx.edges():
        lista_nx.append([e[0], e[1]])
    df_nx=pd.DataFrame(lista_nx)

    g_nk=nk.nxadapter.nx2nk(g_nx)
    lista_nk= []
    for u, v in g_nk.iterEdges():
        lista_nk.append([u, v])
    df_nk=pd.DataFrame(lista_nk)

    g_nk.indexEdges()
    #target_ratio=round(1/(g_nk.numberOfEdges()/g_nk.numberOfNodes()),2)
    target_ratio=0.2
    # Initialize the algorithm
    lds = nk.sparsification.LocalDegreeScore(g_nk)
    # Run
    lds.run()
    # Get edge scores
    ldsScores = lds.scores()

    t0=time.time()
    # Initialize the algorithm
    localDegSparsifier = nk.sparsification.LocalDegreeSparsifier()
    # Get sparsified graph
    localDegGraph = localDegSparsifier.getSparsifiedGraphOfSize(g_nk, target_ratio)
    print('analysis time: ',time.time()-t0,'seconds')

    lista_nk_fil= []
    for u, v in localDegGraph.iterEdges():
        lista_nk_fil.append([u, v])
    df_nk_fil=pd.DataFrame(lista_nk_fil)

    keys=list(df_nk_fil.columns.values)
    i1 = df_nk.set_index(keys).index
    i2 = df_nk_fil.set_index(keys).index
    df_nk_new=df_nk[i1.isin(i2)]
    df_nx_new=df_nx[df_nx.index.isin(df_nk_new.index)]

    keys=list(df_nx_new.columns.values)
    i1a = anno.set_index(['bin1_id','bin2_id']).index
    i1b = anno.set_index(['bin2_id','bin1_id']).index #if the edge storage has been reversed
    i2 = df_nx_new.set_index(keys).index
    anno_new_a=anno[i1a.isin(i2)]
    anno_new_b=anno[i1b.isin(i2)]

    anno_new=pd.concat([anno_new_b,anno_new_a])
    anno_new=anno_new.sort_index()
    pix_new=anno_new[['bin1_id','bin2_id','count']]
    pix_new.to_csv(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/sparse_graphs/weighted_sparsification/{c_id}_count_double_20_sparsification.txt',sep='\t',index=None)
