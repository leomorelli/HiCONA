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
def annotated_bigraph(
    graph:nx.classes.graph.Graph,
    annotation:str,
    bilabel:tuple
):
    g=graph
    if type(graph) != nx.classes.graph.Graph:
        raise TypeError('a networkx graph object is required for these analyses')
    if (type(bilabel) != tuple) | (len(bilabel)!=2):
        raise TypeError('each annotation pair must be provided in a tuple of length = 2')
    try:
        n0=list(g.nodes())[0]
        fmt=g.nodes.data(True)[n0][annotation]
    except KeyError:
        raise KeyError('the name of the annotation provided is not present in nodes annotation. You can check the available annotations with `g.nodes.data(True)[list(g.nodes())[0]][annotation]`')
    # annotate the edge list with terms of the selected annotation
    elist=nx.to_pandas_edgelist(g)
    anno_source=[]
    for n in elist.source.tolist():
        anno_source.append(g.nodes.data()[n][annotation])
    anno_target=[]
    for n in elist.target.tolist():
        anno_target.append(g.nodes.data()[n][annotation])
    elist[f'{annotation}_source']=anno_source
    elist[f'{annotation}_target']=anno_target
    # filter out edges, which nodes does not possess the annotation pair specified in `bilabel`
    elist_filtered=elist[((elist[f'{annotation}_source']==bilabel[0])&(elist[f'{annotation}_target']==bilabel[1]))|((elist[f'{annotation}_source']==bilabel[1])&(elist[f'{annotation}_target']==bilabel[0]))]
    edges_filtered=[]
    for i in elist_filtered.index:
        source=elist_filtered.loc[i,'source']    
        target=elist_filtered.loc[i,'target']
        edges_filtered.append((source,target))
    # bigraph generation
    sg=g.edge_subgraph(edges_filtered)
    return sg

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

def sequential_bins(
    cool_file:Union[cooler.api.Cooler,str],
    chromosome:str
):
    if type(cool_file)==str:
        c=cooler.Cooler(cool_file)
    else:
        c=cool_file
    bins=c.bins()[:]
    to_select=list(bins[bins['chrom']==chromosome].index)
    pix_tot=c.pixels()[:]
    pix=pix_tot[(pix_tot.bin1_id.isin(to_select))&(pix_tot.bin2_id.isin(to_select))]
    for i in range(1):
        if len(pix.index)==0:
            warnings.warn(f'no connections in chromosome {chromosome}')
            continue
        else:
            max_count=max(pix['count'])
            sequential_pixels=list(set(pix['bin1_id'])|set(pix['bin2_id']))
            min_bin=min(sequential_pixels)
            max_bin=max(sequential_pixels)
            sequential_bins=list(bins.index[min_bin:max_bin+1])
            sequential_pairs=[]
            for i in range(len(sequential_bins)):
                if i+1==len(sequential_bins):
                    continue
                n1=sequential_bins[i]
                n2=sequential_bins[i+1]
                sequential_pairs.append((n1,n2))
            df_sequential_pairs=pd.DataFrame(sequential_pairs,columns=['bin1_id','bin2_id'])
            df_sequential_pairs['count']=[max_count for x in range(df_sequential_pairs.shape[0])]
            return df_sequential_pairs

def sequential_pixels(
    cool_file:Union[cooler.api.Cooler,str],
    chromosome:str
):
    if type(cool_file)==str:
        c=cooler.Cooler(cool_file)
    else:
        c=cool_file
    bins=c.bins()[:]
    to_select=list(bins[bins['chrom']==chromosome].index)
    pix_tot=c.pixels()[:]
    pix=pix_tot[(pix_tot.bin1_id.isin(to_select))&(pix_tot.bin2_id.isin(to_select))]
    for i in range(1):
        if len(pix.index)==0:
            warnings.warn(f'no connections in chromosome {chromosome}')
            continue
        else:
            max_count=max(pix['count'])
            sequential_pixels=np.sort(list(set(pix['bin1_id'])|set(pix['bin2_id'])))
            sequential_pairs=[]
            for i in range(len(sequential_pixels)):
                if i+1==len(sequential_pixels):
                    continue
                n1=sequential_pixels[i]
                n2=sequential_pixels[i+1]
                sequential_pairs.append((n1,n2))
            df_sequential_pairs=pd.DataFrame(sequential_pairs,columns=['bin1_id','bin2_id'])
            df_sequential_pairs['count']=[max_count for x in range(df_sequential_pairs.shape[0])]
            return df_sequential_pairs


def linked_graph(
    cool_file:Union[cooler.api.Cooler,str],
    total_graph:Optional[bool]=False
):
    if type(cool_file)==str:
        c=cooler.Cooler(cool_file)
    else:
        c=cool_file
    chromosomes=c.chromnames
    PIX=pd.DataFrame(columns=['bin1_id','bin2_id','count'])
    bins=c.bins()[:]
    pix=c.pixels()[:]
    pix['genomic_link']=[0 for x in range(pix.shape[0])]
    for chrom in chromosomes:
        
        if total_graph==True:
            pix_chr=sequential_bins(c,chrom)
            PIX=pd.concat([PIX,pix_chr])
        else:
            pix_chr=sequential_pixels(c,chrom)
            PIX=pd.concat([PIX,pix_chr])
    PIX['genomic_link']=[1 for x in range(len(PIX.index))]
    pix=pd.concat([pix,PIX])
    pix.index=[x for x in range(len(pix.index))]
    df=cooler.annotate(pix,bins)
    g=nx.from_pandas_edgelist(df,source='bin1_id',target='bin2_id',edge_attr=['count','genomic_link'])
    #node annotation
    node_list=bins[bins.index.isin(np.sort([x for x in g.nodes]))]
    for n in node_list.index:
        g.nodes[n]['chrom'] = node_list['chrom'][n] #chromosomes
        g.nodes[n]['start'] = node_list['start'][n] #bin start
        g.nodes[n]['end'] = node_list['end'][n] #bin start
    # graph attributes (bin size)
    g.graph['bin_size'] = c.binsize
    return g



#8. storing and loading