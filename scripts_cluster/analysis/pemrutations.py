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
import glob
from random import sample
import networkx as nx
import pybedtools



def permutation(
    graph,
    annotation:str,
    terms:Union[list,str,int,float],
    n_perm:Optional[int]=1000,
    stat:Optional[str]='average_degree',
    plot:Optional[bool]=True,
    store_plot:Optional[bool]=False,
    plot_file:Optional[str]='plot.pdf',   #format always pdf
    dpi:Optional[int]=200
):
    g=graph
    if (type(terms)==str)|(type(terms)==int)|(type(terms)==float):
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



import networkx as nx
import graph_tool.all as gt

def get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    if isinstance(key, str):
        # Encode the key as utf-8
        key = key.encode('utf-8', errors='replace')

    # Deal with the value
    if isinstance(value, bool):
        tname = 'bool'

    elif isinstance(value, int):
        tname = 'float'
        value = float(value)

    elif isinstance(value, float):
        tname = 'float'

    elif isinstance(value, str):
        tname = 'string'
        value = value.encode('utf-8', errors='replace')

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)

    #If key is a byte value, decode it to string
    try:
        key = key.decode('utf-8')
    except AttributeError:
        pass

    return tname, value, key


def make_node_df(G):
    nodes = {}
    for node, attribute in G.nodes(data=True):
        if not nodes.get('node'):
            nodes['node'] = [node]
        else:
            nodes['node'].append(node)
        for key, value in attribute.items():
            if not nodes.get(key):
                nodes[key] = [value]
            else:
                nodes[key].append(value)
    df=pd.DataFrame(nodes)
    try:
        df=df[['chrom', 'start', 'end','node']]
    except KeyError:
        df=0
    return df

def filter_nodes(
    graph: nx.classes.graph.Graph,
    method: str='categorical',   # 'label': select nodes characterized by a specific label; 'value': select nodes with an intrinsic value higher lower or euqal to a throeshold'
    annotation: str='',
    labels: Optional[list]=[],
    operation: Optional[str]= 'higher',  #higher, lower, equal, boundaries (range)
    threshold: Optional[Union[int,float]]=0,
    boundaries: Optional[tuple]=()
):
    if len(annotation)<1:
        raise ValueError('A proper annotation must be selected in order to compute the filtering: you can check the available attributes for each node with `g.nodes.data(True)`')
    if method not in ['categorical','numerical']:
        raise KeyError('the filtering can be performed using `categorical` method or `numerical` method')
    if operation not in ['higher','lower','equal','boundaries']:
        raise KeyError('The only operation available are `higher`,`lower`,`equal` or `boundaries`')
    if type(graph) != nx.classes.graph.Graph:
        raise TypeError('a networkx graph object is required for these analyses')
    g=graph
    if method=='categorical':
        selected_nodes=[n for n,v in g.nodes(data=True) if v[annotation] in labels]
        sg = g.subgraph(selected_nodes)
    elif method=='numerical':
        if operation=='higher':
            selected_nodes=[n for n,v in g.nodes(data=True) if v[annotation] >= threshold]
        elif operation=='lower':
            selected_nodes=[n for n,v in g.nodes(data=True) if v[annotation] < threshold]
        elif operation=='equal':
            selected_nodes=[n for n,v in g.nodes(data=True) if v[annotation] == threshold]
        elif operation=='boundaries':
            selected_nodes=[n for n,v in g.nodes(data=True) if boundaries[0] <= v[annotation] < boundaries[1]]
        sg = g.subgraph(selected_nodes)
    return sg

pvals=dict()
annos=['boundaries_Strong', 'boundaries_Weak', 'boundaries_tot','chiptop_H3K27ac', 'chiptop_H3K27me3', 'chiptop_H3K4me1','chiptop_H3K4me2', 'chiptop_H3K4me3', 'chiptop_H3K9ac','chiptop_H3K9me3', 'chiptop_H2A.Z', 'compartments_positive','compartments_negative', 'repetitive_elements_CGi_hg38']
stat='average_degree'


for frac in ['02','05']:
    for cells in [f'GM_count_local_deg_0001_ratio{frac}.txt',f'IMR90_count_local_deg_0005_ratio{frac}.txt',f'HMEC_count_local_deg_01_ratio{frac}.txt',f'HUVEC_count_local_deg_05_ratio{frac}.txt']:
        cell=cells.split('_')[0]
        g=gt.load_graph(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/sparse_graphs/local_degree/g_{cell}_local_deg{frac}.xml.gz')
        for anno in annos:
            gt.graph_draw(g,vertex_fill_color=g.vp[name],output=f'results/permutations/graphs/local_degree/{cell}_local_deg{frac}_{anno}.pdf')
            pval=permutation(g,annotation=anno,terms=[1.0],n_perm=1000,stat=stat, store_plot=True,plot_file=f'results/permutations/plots/local_degree/{cell}_local_deg{frac}_{stat}_{anno}.pdf')
            pvals[f'{anno}'] = pvals.get(f'{anno}', pval)
    df=pd.DataFrame(pvals)
    df.to_csv(f'results/permutations/pvals/local_degree/{cell}_local_deg{frac}_{stat}.txt',sep='\t')
    
    
for frac in ['02','05']:
    for cells in [f'GM_count_local_deg_0001_ratio{frac}.txt',f'IMR90_count_local_deg_0005_ratio{frac}.txt',f'HMEC_count_local_deg_01_ratio{frac}.txt',f'HUVEC_count_local_deg_05_ratio{frac}.txt']:
        cell=cells.split('_')[0]
        g=gt.load_graph(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/sparse_graphs/local_degree/g_{cell}_local_deg{frac}.xml.gz')
        for anno in annos:
            gt.graph_draw(g,vertex_fill_color=g.vp[name],output=f'results/permutations/graphs/local_degree/{cell}_local_deg{frac}_{anno}.pdf')
            pval=permutation(g,annotation=anno,terms=[1.0],n_perm=1000,stat=stat, store_plot=True,plot_file=f'results/permutations/plots/local_degree/{cell}_local_deg{frac}_{stat}_{anno}.pdf')
            pvals[f'{anno}'] = pvals.get(f'{anno}', pval)
        df=pd.DataFrame(pvals)
        df.to_csv(f'results/permutations/pvals/local_degree/{cell}_local_deg{frac}_{stat}.txt',sep='\t')
        
for cells in [f'GM_count',f'IMR90_count',f'HMEC_count',f'HUVEC_count']:
    cell=cells.split('_')[0]
    g=gt.load_graph(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/sparse_graphs/weighted_sparsification/g_{cell}_weighted.xml.gz')
    for anno in annos:
        gt.graph_draw(g,vertex_fill_color=g.vp[name],output=f'results/permutations/graphs/weigthed_sparsification/{cell}_weigthed_sparsification_{anno}.pdf')
        pval=permutation(g,annotation=anno,terms=[1.0],n_perm=1000,stat=stat, store_plot=True,plot_file=f'results/permutations/plots/weigthed_sparsification/{cell}_weigthed_sparsification_{stat}_{anno}.pdf')
        pvals[f'{anno}'] = pvals.get(f'{anno}', pval)
    df=pd.DataFrame(pvals)
    df.to_csv(f'results/permutations/pvals/weigthed_sparsification/{cell}_weigthed_sparsification_{stat}.txt',sep='\t')
