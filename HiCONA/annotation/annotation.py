from typing import Optional, Tuple, Sequence, Type, Union, Literal
from pathlib import Path, PurePath
import warnings
import pandas as pd
import cooler
import pybedtools
import h5py
import numpy as np
import networkx as nx




def parse_annotations(
    filename: Union[Path, str],
    delimiter: Optional[str] = '\t',
    header: Optional = 'infer',
    multi_label: Optional[bool] = False,
    annotated_column: Optional[int] = 999
):
    input_file=Path(filename)      # allow passing strings
    anno=pd.read_table(input_file,header=header,delimiter=delimiter)
    if len(anno.columns)<4:   #default bed file   chr | start | end
        bed_format='simple'     # no annotation provided besides bed regions
    else:
        if multi_label==False:
            bed_format='double'  #each bed regions is annotated with a specific label (i.e. promoters' name)
        else:
            bed_format='multi_label'  #each bed regions is annotated with a specific class (i.e. promoter, enhancer, insulator)
        if annotated_column==999:
            warnings.warn("you have not specified the column with annotation label. The 4th column will be selected by default")
    return bed_format





def cool_processing(
    cool_file
):
    cool_input=cooler.Cooler(cool_file)
    bins = cool_input.bins()[:]  # fetch all the bins
    bins=bins.iloc[:,:3]
    pix = cool_input.pixels()[:]  # fetch all pixels 
    cool_pairs=cooler.annotate(pix, bins)  # chrom1|start1|end1|annotation1|chrom2|start2|end2|annotation2|bin1_id|bin2_id|count
    to_keep=list(set(cool_pairs['bin1_id'])|set(cool_pairs['bin2_id']))
    hic_bed=bins.iloc[to_keep]
    hic_bed.loc[:,'codes']=hic_bed.index               # chrom|start|end|codes
    # adapt format of chromosomes to bedtools
    if str(hic_bed['chrom'].tolist()[0]).startswith('chr')==False:
        chromosomes=['chr'+str(x) for x in hic_bed['chrom'].tolist()]
        hic_bed['chrom']=chromosomes
    intersection_bed= pybedtools.BedTool.from_dataframe(hic_bed)
    return cool_input,intersection_bed





def simple_annotation(
    cool_file: Union[Path, str],
    anno_file: Union[Path, str],
    name: Optional[str]='annotation'
):
    # RESULT: list of int associated to each bin in cool_file.bin()[:] -> 0 if region doesn't show intersection, 1 if it does show intersection with the annotated file
    input_file=Path(cool_file)
    input_anno=Path(anno_file)
    anno_bed=pybedtools.BedTool(input_anno)
    cool_input,intersection_bed= cool_processing(cool_file)
    # 1) annotation generation
    intersected=intersection_bed.intersect(anno_bed)
    intersected_df=intersected.to_dataframe()
    cols=intersected_df.columns
    anno=np.zeros(cool_input.bins()[:].shape[0])
    for i in intersected_df[cols[-1]].tolist():   #intersected_df[cols[-1]]=list of bin codes showing intersection with annotation file
        anno[i]=1
    # 2) storing the list in the cool file
    f = h5py.File(input_file, "a")
    f['bins'].create_dataset(name=name,data=anno)
    f.close()
    cool_output=cooler.Cooler(cool_file)
    return cool_output





def double_annotation(
    cool_file: Union[Path, str],
    anno_file: Union[Path, str],
    name: Optional[str]='annotation',
    annotated_column: Optional[int] = 3
):
    # RESULT: list of int associated to each bin in cool_file.bin()[:] -> 0 if region doesn't show intersection, 1 if it does show intersection with the annotated file
    input_file=Path(cool_file)
    input_anno=Path(anno_file)
    anno_bed=pybedtools.BedTool(input_anno)
    cool_input,intersection_bed= cool_processing(cool_file)
    # 1) annotation generation
    intersected=intersection_bed.intersect(anno_bed,wa=True,wb=True)
    intersected_df_tot=intersected.to_dataframe()
    codes_columns=3
    intersected_df=pd.DataFrame([intersected_df_tot.iloc[:,codes_columns],intersected_df_tot.iloc[:,codes_columns+1+annotated_column]],index=['code','anno_name']).T
    ### 1.1) simple annotation
    anno=np.zeros(cool_input.bins()[:].shape[0])
    for i in intersected_df['code'].tolist():   #intersected_df[cols[-1]]=list of bin codes showing intersection with annotation file
        anno[int(i)]=1
    ### 1.2) annotation names
    anno_names=['0' for x in range(cool_input.bins()[:].shape[0])]
    for cd in set(intersected_df.iloc[:,0]):
        annotated_code=intersected_df[intersected_df['code']==cd]
        code_anno_name=[x for x in annotated_code['anno_name']]
        if len(code_anno_name)==1:
            anno_name=str(code_anno_name[0])
        else:
            anno_name=str(code_anno_name[0])
            for i in range(1,len(code_anno_name)):
                anno_name=str(anno_name)+' '+str(code_anno_name[i])
        anno_names[int(cd)]=anno_name
    asciiList = [n.encode("ascii", "ignore") for n in anno_names]   #format for string supported by h5py
    # 2) storing the list in the cool file
    f = h5py.File(input_file, "a")
    f['bins'].create_dataset(name=name,data=anno)
    f['bins'].create_dataset(name=name+'_'+'name',data=asciiList)
    f.close()
    cool_output=cooler.Cooler(cool_file)
    return cool_output





def hierarchical_annotation(
    cool_file: Union[Path, str],
    anno_file: Union[Path, str],
    hierarchy: Union[Path, str],
    name: Optional[str]='annotation',
    annotated_column: Optional[int] = 3
):
    input_file=Path(cool_file)
    input_anno=Path(anno_file)
    hierarchy=pd.read_table(hierarchy,header=None)
    anno_bed=pybedtools.BedTool(input_anno)
    cool_input,intersection_bed= cool_processing(cool_file)
    # 1) annotation generation
    intersected=intersection_bed.intersect(anno_bed,wa=True,wb=True)
    intersected_df_tot=intersected.to_dataframe()
    codes_columns=3
    intersected_df=pd.DataFrame([intersected_df_tot.iloc[:,codes_columns],intersected_df_tot.iloc[:,codes_columns+1+annotated_column]],index=['code','anno_name']).T
    anno=np.zeros(cool_input.bins()[:].shape[0])
    for i in range(len(hierarchy[0])):
        anno_id=i+1
        anno_term=hierarchy[0][i]
        intersected_df_i=intersected_df[intersected_df['anno_name']==anno_term]
        for cd in intersected_df_i['code'].tolist():   #intersected_df[cols[-1]]=list of bin codes showing intersection with annotation file
            if anno[int(cd)] == 0:
                anno[int(cd)]=anno_id
    # 2) storing the list in the cool file
    f = h5py.File(input_file, "a")
    f['bins'].create_dataset(name=name,data=anno)
    f['bins'][name].attrs[f'{name}_categories']=[x for x in hierarchy[0]]
    f.close()
    cool_output=cooler.Cooler(cool_file)
    return cool_output



def annotation_report(
    cool_file: Union[Path, str],
    hierarchy: Optional[str]='',
    strategy:Optional[str]='simple',
    store_report: Optional[bool] = False,
    report_location: Optional[str]='.'
):
    cool_output=cooler.Cooler(cool_file)
    bins=cool_output.bins()[:]
    pix=cool_output.pixels()[:]
    idxs=list(set(pix['bin1_id'])|set(pix['bin2_id']))
    df=bins.iloc[idxs][strategy]
    df.index=[x for x in range(df.shape[0])]
    tot_nnz_bins=len(df.index)
    not_anno=len(df[df==0])
    print(f''' {round((not_anno/tot_nnz_bins)*100,2)}% ({not_anno}) of Hi-C bins are NOT annotated \n {100 - round((not_anno/tot_nnz_bins)*100,2)}% ({tot_nnz_bins-not_anno}) of Hi-C bins overlap an annotated region  \n''')
    if store_report==True:
        report_name=cool_file.split('/')[-1].split('.')[0]    #name of the file without its extension
        with open(f'{report_location}/{report_name}_report.txt', 'a') as f:
            f.write(f'{round((not_anno/tot_nnz_bins)*100,2)}% ({not_anno}) of Hi-C bins are NOT annotated')
            f.write('\n')
            f.write(f'{100 - round((not_anno/tot_nnz_bins)*100,2)}% ({tot_nnz_bins-not_anno}) of Hi-C bins overlap an annotated region')
            f.write('\n')
            f.write('\n')
            f.close()
    if strategy=='multiple':
        hierarchy_path='../input_files/test/hierarchy.txt'
        hierarchy=pd.read_table(hierarchy_path,header=None)
        for i in range(len(hierarchy[0])):
            anno_term=hierarchy[0][i]
            anno_id=i+1
            n_anno=len(df[df==anno_id])
            print(f' {round((n_anno/tot_nnz_bins)*100,2)}% ({n_anno}) of Hi-C bins are annotated as {anno_term}')
            if store_report==True:
                with open(f'{report_location}/{report_name}_report.txt', 'a') as f:
                    f.write(f'{round((n_anno/tot_nnz_bins)*100,2)}% ({n_anno}) of Hi-C bins are annotated as {anno_term}')
                    f.write('\n')
                    f.close()





def bin_annotation(
    cool_file: Union[Path, str],
    anno_file: Union[Path, str],
    hierarchy: Optional[str]='',
    strategy:Optional[str]='simple',
    name: Optional[str]='annotation',
    annotated_column: Optional[int] = 3,
    report: Optional[bool] = True,
    store_report: Optional[bool] = False,
    report_location: Optional[str]='.'
):
    if strategy=='simple':
        cool_output=simple_annotation(cool_file=cool_file,anno_file=anno_file,name=name)
    elif strategy=='double':
        cool_output=double_annotation(cool_file=cool_file,anno_file=anno_file,name=name,annotated_column=annotated_column)
    elif strategy=='multiple':
        if len(hierarchy)<1:
            raise ValueError('In order to annotate Hi-C bins according to different classes of annotations, you must provide a text file representing the hierarchy of annotation')
        cool_output=hierarchical_annotation(cool_file=cool_file,anno_file=anno_file,hierarchy=hierarchy,name=name,annotated_column=annotated_column)
    else:
        raise ValueError('Remember to provide a suitable entry for the `strategy` of annotation parameter. You can choose one of the following: `simple`, `double`, `multiple`')
    if report==True:
        annotation_report(cool_file=cool_file,hierarchy=hierarchy,strategy=strategy,store_report=store_report)
    return cool_output
    
    
    
def delete_annotation(
    cool_file: Union[Path, str],
    name: Optional[str]='annotation'
):
    f = h5py.File(cool_file, "a")
    del f['bins'][name]
    f.close()
    



def to_bed(
    graph: nx.classes.graph.Graph,
    store_bed: Optional[bool]=False,
    file_name: Optional[str]='./node_annotation.bed',
    delimiter:Optional[str]='\t',
    report: Optional[bool]=False,
    annotation: Optional[str]='annotation',
    store_report:Optional[bool]=False,
    normalize_report:Optional[bool]=True,
):
    if type(graph) != nx.classes.graph.Graph:
        raise TypeError('a networkx graph object is required for these analyses')
    g=graph
    df=pd.DataFrame.from_dict(dict(g.nodes(data=True)), orient='index')
    if report==True:
        report_file=df.value_counts(subset=annotation,normalize=normalize_report)
        if store_report==True:
            file_output=file_name.split('/')
            report_tot=file_output[-1].split('.')
            path='/'.join(file_output[:-1])
            name=report_tot[0]+'_report.'+report_tot[-1]
            if len(path)<1:
                report_output='./'+name
            else:
                report_output=path+'/'+name
            report_file.to_csv(report_output,sep=delimiter)
    if (store_bed==True)|((store_bed==False)&(store_report==True)):
        if file_name=='./node_annotation.bed':
            warnings.warn('it seems that you have not specified the file name with the `file_name` parameter: your data will be stored as `./node_annotation.bed`')
        df.to_csv(file_name,header=None,sep=delimiter)
    return df
    
    
    