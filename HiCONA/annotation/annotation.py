from typing import Optional, Tuple, Sequence, Type, Union, Literal
from pathlib import Path, PurePath
import warnings
import pandas as pd
import cooler
import pybedtools
import h5py
import numpy as np





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
    f.close()
    cool_output=cooler.Cooler(cool_file)
    return cool_output





def bin_annotation(
    cool_file: Union[Path, str],
    anno_file: Union[Path, str],
    annotation_strategy:Optional[str]='simple'|'double'|'hierarchical',
    hierarchy: Optional[Path, str],
    name: Optional[str]='annotation',
    annotated_column: Optional[int] = 3,
    report: Optional[bool] = False,
    save_report: Optional[bool] = False
):
    print(annotation_strategy)
    
    
    
state='hierarchical'
bins=cool_output.bins()[:]
pix=cool_output.pixels()[:]
idxs=list(set(pix['bin1_id'])|set(pix['bin2_id']))
bins.iloc[idxs][state]
    