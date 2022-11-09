from typing import Optional, Tuple, Sequence, Type, Union, Literal
from pathlib import Path, PurePath
import numpy as np
import warnings
import graph_tool.all as gt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import logging
from random import sample

def permutation(
    graph,
    annotation:str,
    terms:Union[list,str],
    n_perm:Optional[int]=1000,
    stat:Optional[str]='average_degree',
    plot:Optional[bool]=True,
    store_plot:Optional[bool]=False,
    plot_file:Optional[str]='plot.pdf',   #format always pdf
    dpi:Optional[int]=200
):
    g=graph
    if type(terms)==str:
        terms=[terms]
    elif type(terms)==list:
        terms=terms
    else:
        raise TypeError('the input for `terms` parameter must be of tipe `list` or `string`')

    anno_list=[x for x in g.vp[f'{annotation}']]
    
    anno_nodes=[index for (index, item) in enumerate(anno_list) if item in terms]
    
    g_real=g.copy()
    g_real.remove_vertex(anno_nodes, fast=True)
    if stat=='average_degree':
        stat_real=gt.vertex_average(g_real,'total')[0]
    elif stat=='clustering_coeff':
        stat_real=gt.global_clustering(g_real)[0]
    elif stat=='central_point_dominance':
        vp, ep = gt.betweenness(g_real)
        stat_real=gt.central_point_dominance(g_real, vp)
    else:
        raise ValueError('please choose a proper statistics among `average_degree`, `clustering_coeff` and `central_point_dominance`')
    vertices=list(g.iter_vertices())

    sample_size=len(anno_nodes)
    distribution=[]
    
    t0=time.time()
    for i in range(n_perm):
        g_perm=g.copy()
        to_remove=sample(vertices,sample_size)
        g_perm.remove_vertex(to_remove, fast=True)
        if stat=='average_degree':
            stat_perm=gt.vertex_average(g_perm,'total')[0]
        elif stat=='clustering_coeff':
            stat_perm=gt.global_clustering(g_perm)[0]
        elif stat=='central_point_dominance':
            vp, ep = gt.betweenness(g_real)
            stat_perm=gt.central_point_dominance(g_perm, vp)
        distribution.append(stat_perm)
        if i==int(n_perm/10):
            t1=time.time()
            warnings.warn(f'1/10 of permutations performed in {t1-t0} seconds. This run will last roughly {(t1-t0)*10} seconds')
    
    if plot==True:
        df=pd.DataFrame(distribution,columns=[f'{stat}'])
        sns.displot(data=df, x=f"{stat}",common_norm=True,kde=True,multiple="stack",color='#C000C0')
        plt.axvline(x=stat_real,color='black')
        if store_plot==True:
            plt.savefig(plot_file,dpi=200)
            
    pval=(len([x for x in distribution if x <stat_real])+1)/(len(distribution)+1)
    return pval
