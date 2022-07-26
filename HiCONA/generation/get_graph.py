from typing import Optional, Tuple, Sequence, Type, Union, Literal
from pathlib import Path, PurePath
import cooler
import h5py
import numpy as np
import warnings
import networkx as nx
import pandas as pd

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
    #node annotation
    node_list=bins[bins.index.isin(np.sort([x for x in g.nodes]))]
    for n in node_list.index:
        g.nodes[n]['chrom'] = node_list['chrom'][n] #chromosomes
        g.nodes[n]['start'] = node_list['start'][n] #bin start
        g.nodes[n]['end'] = node_list['end'][n] #bin start
    # graph attributes (bin size)
    g.graph['bin_size'] = c.binsize
    #if you want to annotate nodes with other annotations beside chromosomes and start
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


# 5. slicing (given 2 regions)

def slice_graph(
    graph: nx.classes.graph.Graph,
    chromosome: Union[str,int],
    start: int,
    end: int
):
    if type(graph) != nx.classes.graph.Graph:
        raise TypeError('a networkx graph object is required for these analyses')
    n0=list(graph.nodes())[0]
    fmt=graph.nodes.data(True)[n0]['chrom']
    if fmt.startswith('chr') != str(chromosome).startswith('chr'):
        warnings.warn("You may have provided the wrong chromosome name format (i.e. chr8 in stead of 8 or viceversa). You can check the chromosome name format that you need with `graph.nodes.data(True)[list(graph.nodes())[0]]['chrom']`")
    # 1. chromosome filtering
    selected_nodes_chrom=[n for n,v in graph.nodes(data=True) if v['chrom'] in [chromosome]]     
    sg_chrom = graph.subgraph(selected_nodes_chrom)
    # 2. region slicing
    selected_nodes_slice=[n for n,v in sg_chrom.nodes(data=True) if start <= v['start'] <= end]
    sg_sliced=sg_chrom.subgraph(selected_nodes_slice)
    return sg_sliced
# 6. iCD given a region (list of region)

def iCDs_from_list(
    graph: nx.classes.graph.Graph,
    list_of_regions: list
):
    g=graph
    df_icd=pd.DataFrame()
    for region in list_of_regions:
        chromosome=region.split('-')[0]
        start=region.split('-')[1]
        end=region.split('-')[2]
        g_df=pd.DataFrame.from_dict(dict(g.nodes(data=True)), orient='index')
        g_df_chrom=g_df[g_df['chrom']==chromosome]
        g_df1=g_df_chrom[(g_df_chrom['start']>=int(start))&(g_df_chrom['end']<=int(end))]
        g_df2=g_df_chrom[(g_df_chrom['start']<=int(start))&(g_df_chrom['end']>=int(start))]
        g_df3=g_df_chrom[(g_df_chrom['start']<=int(end))&(g_df_chrom['end']>=int(end))]
        df_icd=pd.concat([df_icd,g_df1,g_df2,g_df3])
    iCDs=[]
    for n in df_icd.index:
        icd_n=list(nx.node_connected_component(g, n))
        iCDs.append(icd_n)
    # first filter
    iCDs_filtered=list(set(tuple(i) for i in iCDs))
    iCDs_graphs=[]
    for icd in iCDs_filtered:
        selected_nodes=[n for n,v in g.nodes(data=True) if n in icd]
        sg = g.subgraph(selected_nodes)
        iCDs_graphs.append(sg)
    #second filter
    iCDs_graphs_filtered=list(set(tuple(i) for i in iCDs_graphs))
    iCDs_tot=[]
    for icd in iCDs_graphs_filtered:
        selected_nodes=[n for n,v in g.nodes(data=True) if n in icd]
        sg = g.subgraph(selected_nodes)
        iCDs_tot.append(sg)
    return iCDs_tot


def iCDs(
    graph: nx.classes.graph.Graph,
    regions: Union[str,list],
    delimiter: Optional[str]='\t'
):
    g=graph
    if type(g) != nx.classes.graph.Graph:
        raise TypeError('a networkx graph object is required for these analyses')
    if type(regions)==list:
        if len(regions[0].split('-'))<3:
            raise ValueError('Wrong bin format: remember `chr-start-end`')
        list_of_regions=regions
    elif type(regions)==str:
        if len(regions.split('.'))<2:
            if len(regions.split('-'))<3:
                raise ValueError('Wrong bin format: remember `chr-start-end`')
            n0=list(g.nodes())[0]
            fmt=g.nodes.data(True)[n0]['chrom']
            if fmt.startswith('chr') != str(regions).startswith('chr'):
                warnings.warn("You may have provided the wrong chromosome name format (i.e. chr8 in stead of 8 or viceversa). You can check the chromosome name format that you need with `graph.nodes.data(True)[list(graph.nodes())[0]]['chrom']`")
            list_of_regions=[regions]
        else:
            df_regions=pd.read_table(regions,header=None,delimiter=delimiter)
            if len(df_regions.columns)<3:
                raise ValueError('your file must contain the following values for each line: chr, start, end. If the file has been built correctly, you may have selected the wrong delimiter (check the `delimiter` parameter)')
            df_regions['bins']=df_regions[0]+'-'+df_regions[1].astype(str)+'-'+df_regions[2].astype(str)
            list_of_regions=df_regions['bins'].tolist()
    else:
        raise ValueError('The function cannot recognize the regions file')
    iCD_graphs=iCDs_from_list(graph=g,list_of_regions=list_of_regions)
    return iCD_graphs

#7. total linked and linked


#8. storing and loading