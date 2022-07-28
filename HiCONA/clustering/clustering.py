from typing import Optional, Tuple, Sequence, Type, Union, Literal
from pathlib import Path, PurePath
import numpy as np
import warnings
import graph_tool.all as gt
import matplotlib
import pandas as pd
import time
import logging


def minimize_nested(
    xml_file:Union[Path,str],
    layers: Optional[bool]=False,
    n_iter:Optional[int]=100
):
    logging.info('cluster analysis might take a while, in order to reduce its the computation time, you may decrease the number of iteration, with the `n_iter` parameter')
    try:
        g = gt.load_graph(xml_file,fmt='xml')
    except OSError:
        raise OSError('Wrong graph format!')
    if layers==True:
        try:
            g.ep.genomic_link
        except KeyError:
            raise KeyError('in order to perform the layered cluster analysis you have to provide a linked graph')
        min_entropy=10**10
        for i in range(100):
            state = gt.minimize_nested_blockmodel_dl(g,state_args=dict(base_type=gt.LayeredBlockState,state_args=dict(ec=g.ep.genomic_link, layers=True)))
            if state.entropy()<min_entropy:
                min_entropy=state.entropy()
                state_final=state
    elif layers==False:
        min_entropy=10**10
        for i in range(100):
            state = gt.minimize_nested_blockmodel_dl(g)
            if state.entropy()<min_entropy:
                min_entropy=state.entropy()
                state_final=state
    return state_final


def collect_marginal_probabilities(
    graph,
    state,
    forced_iteration:Optional[int]=1000
):
    warnings.warn('this step could be really time consuming')
    g=graph
    time1=time.time()
    gt.mcmc_equilibrate(state, wait=1000, mcmc_args=dict(niter=10))
    time2=time.time()
    logging.info(f'1/2 equilibration performed in {time2-time1}') 
    pv = None
    for i in range(forced_iteration):
        ret = state.mcmc_sweep(niter=10)
        pv = state.get_levels()[0].collect_vertex_marginals(pv)
    time3=time.time()
    logging.info(f'2/2 equilibration performed in {time3-time2}')
    return state, pv


def graph_of_clusters(
    graph,
    state,
    store:Optional[bool]=False,
    path:Optional[str]='.',
    name:Optional[str]='graph_cluster'
):
    g=graph
    if path.endswith('/'):
        path=path[:-1]
    vprop = g.new_vertex_property("double")
    g.vp.cluster = vprop 
    for i in range(len(state.get_bs()[0])):
        g.vp.cluster[i] =state.get_bs()[0][i]
    clusters=[]
    for cl in set(g.vp.cluster.a):
        u = gt.GraphView(g, vfilt=g.vp.cluster.a==cl)
        u = gt.Graph(u, prune=True)
        clusters.append(u)
        if store==True:
            u.save(f"{path}/{name}_{cl}.xml.gz")
    return clusters

