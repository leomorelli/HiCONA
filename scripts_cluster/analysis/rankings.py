import graph_tool.all as gt
import numpy as np
import matplotlib.pyplot as plt
from random import sample
import pandas as pd

#closeness
for THR in [0.05,0.01]:
    for cell in ['GM','IMR90','HMEC','HUVEC']:
        for frac in ['02','05']:
            g=gt.load_graph(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/sparse_graphs/local_degree/g_{cell}_local_deg{frac}.xml.gz')
            centrality=gt.closeness(g,weight=g.ep['count'])
            DF=pd.DataFrame()
            centrality.a=np.nan_to_num(centrality.a)
            vertices=list(g.iter_vertices())
            thr=int(len([x for x in g.vp['_graphml_vertex_id']])*THR)
            top_idxs = centrality.a.argsort()[-thr:][::-1]
            str_thr=str(THR).split('.')[-1]
            sample_size=len(top_idxs)
            annos=['boundaries_Strong', 'boundaries_Weak', 'boundaries_tot','chiptop_H3K27ac', 'chiptop_H3K27me3', 'chiptop_H3K4me1','chiptop_H3K4me2', 'chiptop_H3K4me3', 'chiptop_H3K9ac','chiptop_H3K9me3', 'chiptop_H2A.Z', 'compartments_positive','compartments_negative', 'repetitive_elements_CGi_hg38','repetitive_elements_hg38_SINE','repetitive_elements_hg38_RNA','repetitive_elements_hg38_RC','repetitive_elements_hg38_LTR','repetitive_elements_hg38_LINE','repetitive_elements_hg38_DNA']
            for anno in annos:
                anno_score=[]
                for i in top_idxs:
                    anno_score.append(g.vp[anno][i])
                a_score=sum(anno_score)
                r_score=[]
                for i in range(1000):
                    to_pick=sample(vertices,sample_size)
                    random_score=[]
                    for p in to_pick:
                        random_score.append(g.vp[anno][p])
                    r_score.append(sum(random_score))
                pval=(len([x for x in r_score if x >a_score])+1)/(len(r_score)+1)
                fold_change=(sum(anno_score)/len(top_idxs))/(len([x for x in g.vp[anno] if x>0])/len([x for x in g.vp[anno]]))
                df=pd.DataFrame([fold_change,pval],index=['fold_change','pvalues'],columns=[anno]).T
                DF=pd.concat([DF,df])
                print(anno,': pval =',pval,'   |   ','fold_change = ', fold_change)
            DF.to_csv(f'results/centrality_ranking/local_degree/{cell}_local_degree{frac}_closeness_{str_thr}.txt',sep='\t')

#eigenvector
for THR in [0.05,0.01]:
    for cell in ['GM','IMR90','HMEC','HUVEC']:
        for frac in ['02','05']:
            g=gt.load_graph(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/sparse_graphs/local_degree/g_{cell}_local_deg{frac}.xml.gz')
            max_value,centrality=gt.eigenvector(g,weight=g.ep['count'])
            DF=pd.DataFrame()
            centrality.a=np.nan_to_num(centrality.a)
            vertices=list(g.iter_vertices())
            thr=int(len([x for x in g.vp['_graphml_vertex_id']])*THR)
            top_idxs = centrality.a.argsort()[-thr:][::-1]
            str_thr=str(THR).split('.')[-1]
            sample_size=len(top_idxs)
            annos=['boundaries_Strong', 'boundaries_Weak', 'boundaries_tot','chiptop_H3K27ac', 'chiptop_H3K27me3', 'chiptop_H3K4me1','chiptop_H3K4me2', 'chiptop_H3K4me3', 'chiptop_H3K9ac','chiptop_H3K9me3', 'chiptop_H2A.Z', 'compartments_positive','compartments_negative', 'repetitive_elements_CGi_hg38','repetitive_elements_hg38_SINE','repetitive_elements_hg38_RNA','repetitive_elements_hg38_RC','repetitive_elements_hg38_LTR','repetitive_elements_hg38_LINE','repetitive_elements_hg38_DNA']
            for anno in annos:
                anno_score=[]
                for i in top_idxs:
                    anno_score.append(g.vp[anno][i])
                a_score=sum(anno_score)
                r_score=[]
                for i in range(1000):
                    to_pick=sample(vertices,sample_size)
                    random_score=[]
                    for p in to_pick:
                        random_score.append(g.vp[anno][p])
                    r_score.append(sum(random_score))
                pval=(len([x for x in r_score if x >a_score])+1)/(len(r_score)+1)
                fold_change=(sum(anno_score)/len(top_idxs))/(len([x for x in g.vp[anno] if x>0])/len([x for x in g.vp[anno]]))
                df=pd.DataFrame([fold_change,pval],index=['fold_change','pvalues'],columns=[anno]).T
                DF=pd.concat([DF,df])
                print(anno,': pval =',pval,'   |   ','fold_change = ', fold_change)
            DF.to_csv(f'results/centrality_ranking/local_degree/{cell}_local_degree{frac}_eigenvector_{str_thr}.txt',sep='\t')

#betweenness
for THR in [0.05,0.01]:
    for cell in ['GM','IMR90','HMEC','HUVEC']:
        for frac in ['02','05']:
            g=gt.load_graph(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/sparse_graphs/local_degree/g_{cell}_local_deg{frac}.xml.gz')
            centrality,edges_cen=gt.betweenness(g,weight=g.ep['count'])
            DF=pd.DataFrame()
            centrality.a=np.nan_to_num(centrality.a)
            vertices=list(g.iter_vertices())
            thr=int(len([x for x in g.vp['_graphml_vertex_id']])*THR)
            top_idxs = centrality.a.argsort()[-thr:][::-1]
            str_thr=str(THR).split('.')[-1]
            sample_size=len(top_idxs)
            annos=['boundaries_Strong', 'boundaries_Weak', 'boundaries_tot','chiptop_H3K27ac', 'chiptop_H3K27me3', 'chiptop_H3K4me1','chiptop_H3K4me2', 'chiptop_H3K4me3', 'chiptop_H3K9ac','chiptop_H3K9me3', 'chiptop_H2A.Z', 'compartments_positive','compartments_negative', 'repetitive_elements_CGi_hg38','repetitive_elements_hg38_SINE','repetitive_elements_hg38_RNA','repetitive_elements_hg38_RC','repetitive_elements_hg38_LTR','repetitive_elements_hg38_LINE','repetitive_elements_hg38_DNA']
            for anno in annos:
                anno_score=[]
                for i in top_idxs:
                    anno_score.append(g.vp[anno][i])
                a_score=sum(anno_score)
                r_score=[]
                for i in range(1000):
                    to_pick=sample(vertices,sample_size)
                    random_score=[]
                    for p in to_pick:
                        random_score.append(g.vp[anno][p])
                    r_score.append(sum(random_score))
                pval=(len([x for x in r_score if x >a_score])+1)/(len(r_score)+1)
                fold_change=(sum(anno_score)/len(top_idxs))/(len([x for x in g.vp[anno] if x>0])/len([x for x in g.vp[anno]]))
                df=pd.DataFrame([fold_change,pval],index=['fold_change','pvalues'],columns=[anno]).T
                DF=pd.concat([DF,df])
                print(anno,': pval =',pval,'   |   ','fold_change = ', fold_change)
            DF.to_csv(f'results/centrality_ranking/local_degree/{cell}_local_degree{frac}_betweenness_{str_thr}.txt',sep='\t')

#pagerank
for THR in [0.05,0.01]:
    for cell in ['GM','IMR90','HMEC','HUVEC']:
        for frac in ['02','05']:
            g=gt.load_graph(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/sparse_graphs/local_degree/g_{cell}_local_deg{frac}.xml.gz')
            centrality=gt.pagerank(g,weight=g.ep['count'])
            DF=pd.DataFrame()
            centrality.a=np.nan_to_num(centrality.a)
            vertices=list(g.iter_vertices())
            thr=int(len([x for x in g.vp['_graphml_vertex_id']])*THR)
            top_idxs = centrality.a.argsort()[-thr:][::-1]
            str_thr=str(THR).split('.')[-1]
            sample_size=len(top_idxs)
            annos=['boundaries_Strong', 'boundaries_Weak', 'boundaries_tot','chiptop_H3K27ac', 'chiptop_H3K27me3', 'chiptop_H3K4me1','chiptop_H3K4me2', 'chiptop_H3K4me3', 'chiptop_H3K9ac','chiptop_H3K9me3', 'chiptop_H2A.Z', 'compartments_positive','compartments_negative', 'repetitive_elements_CGi_hg38','repetitive_elements_hg38_SINE','repetitive_elements_hg38_RNA','repetitive_elements_hg38_RC','repetitive_elements_hg38_LTR','repetitive_elements_hg38_LINE','repetitive_elements_hg38_DNA']
            for anno in annos:
                anno_score=[]
                for i in top_idxs:
                    anno_score.append(g.vp[anno][i])
                a_score=sum(anno_score)
                r_score=[]
                for i in range(1000):
                    to_pick=sample(vertices,sample_size)
                    random_score=[]
                    for p in to_pick:
                        random_score.append(g.vp[anno][p])
                    r_score.append(sum(random_score))
                pval=(len([x for x in r_score if x >a_score])+1)/(len(r_score)+1)
                fold_change=(sum(anno_score)/len(top_idxs))/(len([x for x in g.vp[anno] if x>0])/len([x for x in g.vp[anno]]))
                df=pd.DataFrame([fold_change,pval],index=['fold_change','pvalues'],columns=[anno]).T
                DF=pd.concat([DF,df])
                print(anno,': pval =',pval,'   |   ','fold_change = ', fold_change)
            DF.to_csv(f'results/centrality_ranking/local_degree/{cell}_local_degree{frac}_pagerank_{str_thr}.txt',sep='\t')

#clustering coefficient
for THR in [0.05,0.01]:
    for cell in ['GM','IMR90','HMEC','HUVEC']:
        for frac in ['02','05']:
            g=gt.load_graph(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/sparse_graphs/local_degree/g_{cell}_local_deg{frac}.xml.gz')
            centrality=gt.local_clustering(g,weight=g.ep['count'])
            DF=pd.DataFrame()
            centrality.a=np.nan_to_num(centrality.a)
            vertices=list(g.iter_vertices())
            thr=int(len([x for x in g.vp['_graphml_vertex_id']])*THR)
            top_idxs = centrality.a.argsort()[-thr:][::-1]
            str_thr=str(THR).split('.')[-1]
            sample_size=len(top_idxs)
            annos=['boundaries_Strong', 'boundaries_Weak', 'boundaries_tot','chiptop_H3K27ac', 'chiptop_H3K27me3', 'chiptop_H3K4me1','chiptop_H3K4me2', 'chiptop_H3K4me3', 'chiptop_H3K9ac','chiptop_H3K9me3', 'chiptop_H2A.Z', 'compartments_positive','compartments_negative', 'repetitive_elements_CGi_hg38','repetitive_elements_hg38_SINE','repetitive_elements_hg38_RNA','repetitive_elements_hg38_RC','repetitive_elements_hg38_LTR','repetitive_elements_hg38_LINE','repetitive_elements_hg38_DNA']
            for anno in annos:
                anno_score=[]
                for i in top_idxs:
                    anno_score.append(g.vp[anno][i])
                a_score=sum(anno_score)
                r_score=[]
                for i in range(1000):
                    to_pick=sample(vertices,sample_size)
                    random_score=[]
                    for p in to_pick:
                        random_score.append(g.vp[anno][p])
                    r_score.append(sum(random_score))
                pval=(len([x for x in r_score if x >a_score])+1)/(len(r_score)+1)
                fold_change=(sum(anno_score)/len(top_idxs))/(len([x for x in g.vp[anno] if x>0])/len([x for x in g.vp[anno]]))
                df=pd.DataFrame([fold_change,pval],index=['fold_change','pvalues'],columns=[anno]).T
                DF=pd.concat([DF,df])
                print(anno,': pval =',pval,'   |   ','fold_change = ', fold_change)
            DF.to_csv(f'results/centrality_ranking/local_degree/{cell}_local_degree{frac}_local_clustering_{str_thr}.txt',sep='\t')

#degree
for THR in [0.05,0.01]:
    for cell in ['GM','IMR90','HMEC','HUVEC']:
        for frac in ['02','05']:
            g=gt.load_graph(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/sparse_graphs/local_degree/g_{cell}_local_deg{frac}.xml.gz')
            vertices=list(g.iter_vertices())
            centrality=g.get_total_degrees(vertices)
            DF=pd.DataFrame()
            centrality=np.nan_to_num(centrality)
            thr=int(len([x for x in g.vp['_graphml_vertex_id']])*THR)
            top_idxs = centrality.argsort()[-thr:][::-1]
            str_thr=str(THR).split('.')[-1]
            sample_size=len(top_idxs)
            annos=['boundaries_Strong', 'boundaries_Weak', 'boundaries_tot','chiptop_H3K27ac', 'chiptop_H3K27me3', 'chiptop_H3K4me1','chiptop_H3K4me2', 'chiptop_H3K4me3', 'chiptop_H3K9ac','chiptop_H3K9me3', 'chiptop_H2A.Z', 'compartments_positive','compartments_negative', 'repetitive_elements_CGi_hg38','repetitive_elements_hg38_SINE','repetitive_elements_hg38_RNA','repetitive_elements_hg38_RC','repetitive_elements_hg38_LTR','repetitive_elements_hg38_LINE','repetitive_elements_hg38_DNA']
            for anno in annos:
                anno_score=[]
                for i in top_idxs:
                    anno_score.append(g.vp[anno][i])
                a_score=sum(anno_score)
                r_score=[]
                for i in range(1000):
                    to_pick=sample(vertices,sample_size)
                    random_score=[]
                    for p in to_pick:
                        random_score.append(g.vp[anno][p])
                    r_score.append(sum(random_score))
                pval=(len([x for x in r_score if x >a_score])+1)/(len(r_score)+1)
                fold_change=(sum(anno_score)/len(top_idxs))/(len([x for x in g.vp[anno] if x>0])/len([x for x in g.vp[anno]]))
                df=pd.DataFrame([fold_change,pval],index=['fold_change','pvalues'],columns=[anno]).T
                DF=pd.concat([DF,df])
                print(anno,': pval =',pval,'   |   ','fold_change = ', fold_change)
            DF.to_csv(f'results/centrality_ranking/local_degree/{cell}_local_degree{frac}_degree_{str_thr}.txt',sep='\t')
