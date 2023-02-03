import networkx as nx
import cooler 
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
import time
import seaborn as sns

cells=['Rao_2014_GM_MboI_4DNFIXP4QG5B.mcool','Rao_2014_HUVEC_MboI_4DNFIRMZ7QTE.mcool','Rao_2014_HMEC_MboI_4DNFIGUIV5KO.mcool','Rao_2014_IMR90_MboI_4DNFIJTOIGOI.mcool']
exp=['ICE','KR','VC','VC_SQRT']


for cell in cells:
    for e in exp:
        idx=cell.split('_')[2]
        c=cooler.Cooler(f'mcool/{cell}::resolutions/10000')
        pix=c.pixels()[:]
        filtered_df=pd.read_table(f'sparse_graphs/weighted_sparsification/{cell}_{e}_norm.bed')
        pix=pix[pix['count']>1]
        pix=pix[pix['bin1_id']!=pix['bin2_id']]
        filtered_df['code1']=filtered_df['source'].astype(str)+'-'+filtered_df['target'].astype(str)
        filtered_df['code2']=filtered_df['target'].astype(str)+'-'+filtered_df['source'].astype(str)
        pix['code']=pix['bin1_id'].astype(str)+'-'+pix['bin2_id'].astype(str)
        pix_new_a=pix[pix['code'].isin(filtered_df['code1'])]
        pix_new_b=pix[pix['code'].isin(filtered_df['code2'])]
        pix_new=pd.concat([pix_new_b,pix_new_a])
        pix_new=pix_new.sort_index()
        pix_new=pix_new.iloc[:,0:3]
        pix_new.to_csv(f'sparse_graphs/weighted_sparsification/{idx}_{e}_norm_filtered_pix.bed',sep='\t',index=None)
