import graph_tool.all as gt
import warnings
import pandas as pd
import numpy as np
import itertools
import time
import scipy as scp
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
    if len(sig_motifs)==0:
        warnings.warn(f'Zero network motifs seems to be statistically significant. You may try to increase the threshold, which is currently {threshold}')
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
    g,
    annotation,
    matrix_motif,
):
    categories=set([x for x in g.vp[annotation]])
    results=[]
    for mm in matrix_motif:
        pm=mm[0]
        m=mm[1]
        array_m=np.array(m)
        array_m_T=array_m.T
        anno_matrix=[]
        for n in array_m_T:
            anno_node=[]
            for c in categories:
                anno_node.append(list(n).count(c))
            anno_matrix.append(anno_node)
        df=pd.DataFrame(anno_matrix,columns=categories).T
        df=df.sort_index()
        results.append([pm,df])
    return results  # return -> motif(graph),df (annoXnodes)


#return motif graph, text property maps of maximum anno per node, int property map for fraction of max anno
def count_anno(
    anno_nodes
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




# anno sig
def annotation_significance(
    g,
    annotation,
    anno_nodes,
    permutation:Optional[int]=1000
):
    anno_population=[x for x in g.vp[annotation]]
    anno_population_count=pd.Series(anno_population).value_counts().sort_index()
    results=[]
    for pm in anno_nodes:
        pm_object=pm[0]
        anno_n=pm[1]
        motif_sig=[]
        for n in anno_n.columns:
            node_sig=[]
            for a in range(len(anno_population_count)):
                node=anno_n[n].sort_index().tolist()
                anno_count_a=anno_population_count.tolist()
                success=node[a]
                p_success=anno_count_a[a]
                node.pop(a)
                failure=sum(node)
                anno_count_a.pop(a)
                p_failure=sum(anno_count_a)
                node_a=[success,failure]
                p_a=[p_success,p_failure]
                p_a_norm=[x/sum(p_a) for x in p_a]
                prob_real=scp.stats.multinomial.pmf(x=node_a, p=p_a_norm,n=sum(node_a))

                probs_rand=[]
                for x in range(permutation):
                    picks=[]
                    for i in range(sum(node)):
                        idx=np.random.randint(low=2,high=len(anno_population),size=1)
                        picks.append(anno_population[int(idx)])
                    picks_count=pd.Series(picks,dtype='category').value_counts().sort_index()
                    if len(anno_population_count.index)!=len(picks_count.index):
                        lacking_annos=list(set(anno_population_count.index)-set(picks_count.index))
                        series_to_add=pd.Series([0 for a in range(len(lacking_annos))],index=lacking_annos)
                        picks_count=pd.concat([picks_count,series_to_add]).sort_index()
                    node_rand=picks_count.tolist()
                    success=node_rand[a]
                    node_rand.pop(a)
                    failure=sum(node_rand)
                    node_rand_a=[success,failure]
                    p_rand=scp.stats.multinomial.pmf(x=node_rand_a, p=p_a_norm,n=sum(node_rand_a))
                    probs_rand.append(p_rand)
                node_sig.append(len([x for x in probs_rand if x<prob_real])/permutation)
            motif_sig.append(node_sig)
        df=pd.DataFrame(motif_sig,columns=anno_population_count.index).T
        results.append([pm_object,df])
    return results



#link
def motifs_analysis(
    g,
    n_vertices:int,
    annotation:str,
    motifs_sig_threshold:Optional[float]=0.01,
    n_shuffles:Optional[int]=1000,
    analysis:Optional[str]='sig', #can be also `count`
):
    sig_motifs=motifs_sig(g,n_vertices=n_vertices,n_shuffles=n_shuffles,threshold=motifs_sig_threshold)
    pmXnodes=motifs_annotation(g,n_vertices=n_vertices,motifs_list=sig_motifs,annotation=annotation,total_output=False)
    annoXnodes=prop_to_anno(g,annotation=annotation,matrix_motif=pmXnodes)
    if analysis=='sig':
        pm_anno_sig=annotation_significance(g,annotation=annotation,anno_nodes=annoXnodes,permutation=n_shuffles)
        return pm_anno_sig
    elif max_annotation=='count':
        pm_anno_max=count_anno(anno_nodes=annoXnodes)
        return pm_anno_max

