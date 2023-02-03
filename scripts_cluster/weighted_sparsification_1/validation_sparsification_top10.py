import cooler 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import seaborn as sns




for cell in ['IMR90','GM','HUVEC','HMEC']:
    for norm in ['ICE','KR','VC','VC_SQRT']:
        pix=pd.read_table(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/sparse_graphs/weighted_sparsification/{cell}_{norm}_norm_filtered_pix.bed')
        bins=pd.read_table(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/sparse_graphs/weighted_sparsification/bins_annotated_{cell}_top10.bed')
        anno=cooler.annotate(pix,bins)
        bins_comp_neg=bins[bins['compartments_negative']==1]
        bins_comp_pos=bins[bins['compartments_positive']==1]
        anno_pos=cooler.annotate(pix,bins_comp_pos)
        anno_pos=anno_pos.dropna()

        anno_neg=cooler.annotate(pix,bins_comp_neg)
        anno_neg=anno_neg.dropna()

        cols_marks=sorted(list(set([x[:-1] for x in anno.columns if (x.startswith('chip')|x.startswith('bound'))])))

        palette = {
            '+': 'firebrick',
            '-': 'royalblue',
        }
        for m1 in cols_marks:
            anno_mark_neg=anno_neg[(anno_neg[f'{m1}1']==1)|(anno_neg[f'{m1}2']==1)]
            anno_mark_neg.loc[:,'dist']=np.log1p(np.abs(anno_neg['start2']-anno_neg['start1']))
            anno_mark_pos=anno_pos[(anno_pos[f'{m1}1']==1)|(anno_pos[f'{m1}2']==1)]
            anno_mark_pos.loc[:,'dist']=np.log1p(np.abs(anno_pos['start2']-anno_pos['start1']))
            edges_pos=pd.DataFrame()
            edges_neg=pd.DataFrame()
            for m2 in cols_marks:
                edge_pos_m1m2=pd.DataFrame()
                edge_neg_m1m2=pd.DataFrame()
                edge_neg_m1m2[f'count']=np.log1p(anno_mark_neg[((anno_mark_neg[f'{m1}1']==1)&(anno_mark_neg[f'{m2}2']==1))|((anno_mark_neg[f'{m2}1']==1)&(anno_mark_neg[f'{m1}2']==1))]['count'])
                edge_neg_m1m2['mark']=['|'.join(sorted([m1,m2])) for x in range(edge_neg_m1m2.shape[0])]
                edge_neg_m1m2['abundance']=[anno_mark_neg[((anno_mark_neg[f'{m1}1']==1)&(anno_mark_neg[f'{m2}2']==1))|((anno_mark_neg[f'{m2}1']==1)&(anno_mark_neg[f'{m1}2']==1))].shape[0]/anno_mark_neg.shape[0] for x in range(edge_neg_m1m2.shape[0])]
                edge_neg_m1m2['dist']=anno_mark_neg[((anno_mark_neg[f'{m1}1']==1)&(anno_mark_neg[f'{m2}2']==1))|((anno_mark_neg[f'{m2}1']==1)&(anno_mark_neg[f'{m1}2']==1))]['dist']
                edge_pos_m1m2[f'count']=np.log1p(anno_mark_pos[((anno_mark_pos[f'{m1}1']==1)&(anno_mark_pos[f'{m2}2']==1))|((anno_mark_pos[f'{m2}1']==1)&(anno_mark_pos[f'{m1}2']==1))]['count'])
                edge_pos_m1m2['mark']=['|'.join(sorted([m1,m2])) for x in range(edge_pos_m1m2.shape[0])]
                edge_pos_m1m2['abundance']=[anno_mark_pos[((anno_mark_pos[f'{m1}1']==1)&(anno_mark_pos[f'{m2}2']==1))|((anno_mark_pos[f'{m2}1']==1)&(anno_mark_pos[f'{m1}2']==1))].shape[0]/anno_mark_pos.shape[0] for x in range(edge_pos_m1m2.shape[0])]
                edge_pos_m1m2['dist']=anno_mark_pos[((anno_mark_pos[f'{m1}1']==1)&(anno_mark_pos[f'{m2}2']==1))|((anno_mark_pos[f'{m2}1']==1)&(anno_mark_pos[f'{m1}2']==1))]['dist']
                edges_pos=pd.concat([edges_pos,edge_pos_m1m2])
                edges_neg=pd.concat([edges_neg,edge_neg_m1m2])
            edges_neg['comp']=['-' for x in range(edges_neg.shape[0])]
            edges_pos['comp']=['+' for x in range(edges_pos.shape[0])]
            edges=pd.concat([edges_pos,edges_neg])
            plt.figure()
            sns.histplot(edges, x="dist", y="count", hue="comp",palette=palette).set(title=f'{cell} \n Norm: {norm} \n {m1} count_vs_distance')
            plt.savefig(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/results/sparsification_validation/{cell}_{norm}_norm_{m1}_count_vs_distance_top10.pdf')
            plt.figure()
            sns.barplot(edges,x='mark',y='abundance',hue='comp',palette=palette).set(title=f'{cell} \n Norm: {norm} \n {m1}_edges_abundance')
            plt.xticks(rotation=15, ha='right',fontsize=6)
            plt.savefig(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/results/sparsification_validation/{cell}_{norm}_norm_{m1}_edges_abundance_top10.pdf')
            plt.figure()
            sns.violinplot(edges,x='mark',y='count',hue='comp',palette=palette).set(title=f'{cell} \n Norm: {norm} \n {m1}_edges_count')
            plt.xticks(rotation=15, ha='right',fontsize=6)
            plt.savefig(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/results/sparsification_validation/{cell}_{norm}_norm_{m1}_edges_count_top10.pdf')
            plt.figure()
            sns.violinplot(edges,x='mark',y='dist',hue='comp',palette=palette).set(title=f'{cell} \n Norm: {norm} \n {m1}_edges_distance')
            plt.xticks(rotation=15, ha='right',fontsize=6)
            plt.savefig(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/results/sparsification_validation/{cell}_{norm}_norm_{m1}_edges_distance_top10.pdf')
