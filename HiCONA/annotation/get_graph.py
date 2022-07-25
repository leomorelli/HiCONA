from typing import Optional, Tuple, Sequence, Type, Union, Literal
from pathlib import Path, PurePath
import cooler
import h5py
import numpy as np
import warnings
import networkx as nx


#1. generate the total graph

#2 annotate the total graph:
#  -annotation
#  -cc

def get_graph(
    cool_file: Union[cooler.api.Cooler,str],
    annotation: Optional[list]=[]
):
    if type(annotation)!=list:
        raise TypeError('the input for the `annotation` parameter must be of tipe `list`')
    if type(cool_file)==str:
        c=cooler.Cooler(cool_file)
    else:
        c=cool_file
    #edge annotation
    bins=c.bins()[:]
    pix=c.pixels()[:]
    df=cooler.annotate(pix,bins)
    g=nx.from_pandas_edgelist(df,source='bin1_id',target='bin2_id',edge_attr='count')
    #if you want to annotate nodes as well
    if len(annotation)>0:
        for a in annotation:
            try:
                c.bins()[a][0]
                node_list=bins[bins.index.isin(np.sort([x for x in g.nodes]))]
                anno_types=set(node_list[a])
                if len(anno_types)<3:
                    for n in node_list.index:
                        g.nodes[n][a] = node_list[a][n]
                else:
                    try:
                        f = h5py.File(c.uri.split(':')[0], "a")
                        hierarchy=f['bins'][a].attrs[f'{a}_categories']
                        f.close()
                        for n in node_list.index:
                            anno_id=int(node_list[a][n])-1
                            if anno_id==-1:
                                g.nodes[n][a]='nan'
                            else:
                                g.nodes[n][a] = hierarchy[anno_id]
                    except KeyError:
                        warnings.warn("no metadata describing your categories available, using raw code instead")
                        for n in node_list.index:
                            g.nodes[n][a] = node_list[a][n]
            except KeyError:
                raise KeyError(f'`{a}` is not present in the `bins` table. Try `cool_file.bins()[:] in order to take a look to your available annotations`')
    return g


#3. filter out nodes and edges according to annotation or edges according to cc

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


def filter_edges(
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
        selected_edges=[(u, v, e) for u,v,e in g.edges(data=True) if e[annotation] in labels]
        sg = nx.Graph(((u, v, e) for u,v,e in selected_edges))
    elif method=='numerical':
        if operation=='higher':
            selected_edges=[(u,v,e) for u,v,e in g.edges(data=True) if e[annotation] >= threshold]
        elif operation=='lower':
            selected_edges=[(u,v,e) for u,v,e in g.edges(data=True) if e[annotation] < threshold]
        elif operation=='equal':
            selected_edges=[(u,v,e) for u,v,e in g.edges(data=True) if e[annotation] == threshold]
        elif operation=='boundaries':
            selected_edges=[(u,v,e) for u,v,e in g.edges(data=True) if boundaries[0] <= e[annotation]  < boundaries[1]]
        sg = nx.Graph(((u, v, e) for u,v,e in selected_edges))
    return sg

#4. generate PE,PP,EE





#5. storing and loading