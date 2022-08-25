import graph_tool.all as gt
import warnings
import pandas as pd
import numpy as np
import itertools
import time
from typing import Optional, Tuple, Sequence, Type, Union, Literal


# 1. SIGNIFICATIVE MOTIFS

def motifs_sig(
    g,
    n_vertices:int,
    n_shuffles:Optional[int]=1000,
    threshold:Optional[float]=0.01
):
    tot_random_motifs=[]
    sig_motifs=[]
    t0=time.time()
    for i in range(n_shuffles):
        if i == int(n_shuffles/10):
            warnings.warn(f'the analysis would take approximately {(time.time()-t0)*10} seconds more. If you need a faster analysis you can try to reduce n_shuffle parameter (it is currently set at {n_shuffles})')
        analysis=gt.motif_significance(g,n_vertices,n_shuffles=1,full_output=True)
        if i==0:
            real_motifs=[x for x in analysis[2] if x>0]
        random_motifs=list(analysis[3][:len(real_motifs)])
        tot_random_motifs.append(random_motifs)
    tot_random_motifs_T=np.array(tot_random_motifs).T
    for i in range(len(real_motifs)):
        pval=len([x for x in tot_random_motifs_T[i] if x > real_motifs[i]])/n_shuffles
        if pval<=threshold:
            sig_motifs.append((analysis[0][i],pval))
    return sig_motifs



# 2. MOTIFS ANNOTATIONS
#    1. max (text) + fraction_max (size)   -> 1-3 anno X nodes  (dimensions)
#    2. sig (text) + fraction_sig (size)
#    3. pie (sig) + sig (text) + fraction_sig (size)
#    clustering of motifs given the array (knn)   -> property X nodes (dimensions)

def motifs_annotation(
    g,
    n_vertices:int,
    motifs_list:list,
    annotation:str,
    total_output:Optional[bool]=False
):
    try:
        g.vp[annotation]
        annos=[x for x in g.vp[annotation]]
    except KeyError:
        raise KeyError(f'no vertex property map is named as {vertex_text}. Check the available vertex annotation with g.vp')
    results=[]
    motifs,counts,property_maps=gt.motifs(g,n_vertices,motif_list=[motifs_list[i][0] for i in range(len(motifs_list))],return_maps=True)
    for i in range(len(motifs_list)):
        motif=motifs[i]
        pval=motifs_list[i][1]
        pmap=property_maps[i]
        pmap_list=[list(x) for x in pmap]
        pmap_list.sort()
        pmap_filtered=list(pmap_list for pmap_list,_ in itertools.groupby(pmap_list))
        pm_anno=[]
        for j in range(len(pmap_filtered)):
            pm_j=[]
            for x in pmap_filtered[j]:
                pm_j.append([a for a in g.vp[annotation]][x])
            pm_anno.append(pm_j)
        if total_output==True:
            results.append([motif,pm_anno,pval])
        else:
            results.append([motif,pm_anno])
    return results   # return -> motif(graph),array (propertiesXnodes), pvalue


def prop_to_anno(
    matrix_motif,
):
    results=[]
    for mm in matrix_motif:
        pm=mm[0]
        m=mm[1]
        categories=[]
        for i in m:
            categories=list(set(categories)|set(i))
        array_m=np.array(m)
        array_m_T=array_m.T
        anno_matrix=[]
        for n in array_m_T:
            anno_node=[]
            for c in categories:
                anno_node.append(list(n).count(c))
            anno_matrix.append(anno_node)
        df=pd.DataFrame(anno_matrix,columns=categories).T
        results.append([pm,df])
    return results    # return -> motif(graph),df (annoXnodes)


#return motif graph, text property maps of maximum anno per node, int property map for fraction of max anno
def max_anno(
    matrix_anno_motif
):
    results=[]
    for mm in matrix_anno_motif:
        pm=mm[0]
        df=mm[1]
        anno_max=[]
        anno_fraction=[]
        for n in df.columns:
            df_n=df[n]
            anno_max.append(df_n.idxmax())
            anno_fraction.append(max(df_n)/sum(df_n))
        anno_pm=pm.new_vertex_property('string',vals=anno_max)
        fraction_pm=pm.new_vertex_property('float',vals=anno_fraction)
        results.append([pm,anno_pm,fraction_pm])
    return results