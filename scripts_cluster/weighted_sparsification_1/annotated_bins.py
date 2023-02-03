
import cooler 
import pybedtools
import pandas as pd
import numpy as np

def annot(annotation,bins):
    bins.loc[:,'codes']=bins.index
    intersection_bed= pybedtools.BedTool.from_dataframe(bins)
    anno_bed=pybedtools.BedTool(annotation)
    intersected=intersection_bed.intersect(anno_bed)
    intersected_df=intersected.to_dataframe()
    cols=intersected_df.columns
    anno=np.zeros(bins.shape[0])
    for i in intersected_df[cols[-1]].tolist():   #intersected_df[cols[-1]]=list of bin codes showing intersection with annotation file
        anno[i]=1
    return anno

annos=pd.DataFrame()
c=cooler.Cooler('/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/mcool/Rao_2014_HUVEC_MboI_4DNFIRMZ7QTE.mcool::resolutions/10000')

for cell in ['GM12878','IMR90','HMEC','HUVEC']:
    bins=c.bins()[:]
    bins=bins.iloc[:,:3]
    if str(bins['chrom'].tolist()[0]).startswith('chr')==False:
        chromosomes=['chr'+str(x) for x in bins['chrom'].tolist()]
        bins['chrom']=chromosomes
    for type_anno in ['boundaries','chiptop','compartments','repetitive_elements']:
        if type_anno=='boundaries':
            for strength in ['Strong','Weak','tot']:
                anno=annot(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/annotations/{type_anno}/{cell}_boundaries_{strength}.bed',bins)
                annos[f'{type_anno}_{strength}']=anno
        elif type_anno=='chiptop':
            for modification in ['H3K27ac','H3K27me3','H3K4me1','H3K4me2','H3K4me3','H3K9ac','H3K9me3','H2A.Z']:
                if modification=='H3K9me3':
                    anno=annot(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/annotations/{type_anno}/{cell}_{modification}_broadPeak_top5.bed',bins)
                    annos[f'{type_anno}_{modification}']=anno
                else:
                    anno=annot(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/annotations/{type_anno}/{cell}_{modification}_narrowPeak_top5.bed',bins)
                    annos[f'{type_anno}_{modification}']=anno
        elif type_anno=='compartments':
            for comp in ['positive','negative']:
                anno=annot(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/annotations/{type_anno}/{cell}_compartment_{comp}.bed',bins)
                annos[f'{type_anno}_{comp}']=anno
        else:
            for rep in ['CGi_hg38.bed','hg38_DNA.bed','hg38_LINE.bed','hg38_LTR.bed', 'hg38_RC.bed' ,'hg38_RNA.bed','hg38_SINE.bed']:
                anno=annot(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/annotations/{type_anno}/{rep}',bins)
                r=rep.split('.')[0]
                annos[f'{type_anno}_{r}']=anno
        DF=pd.concat([bins.T,annos.T]).T
        DF.to_csv(f'/shares/CIBIO-Storage/GROUPS/sharedCE/Leonardo/HiCONA/Rao_2014/sparse_graphs/weighted_sparsification/bins_annotated_{cell}.bed',sep='\t',index=None)
