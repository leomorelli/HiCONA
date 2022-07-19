from typing import Optional, Tuple, Sequence, Type, Union, Literal
from pathlib import Path, PurePath
import warnings
import subprocess
import pandas as pd

text_exts = {
    'csv',
    'tsv',
    'bed',
    'bedpe',
    'coo',
    'ginteractions',
    'txt',  # these four are all equivalent
}
avail_exts = {
    'h5',
    'hicpro',
    'h5ad',
    'hic',
    'cool',
} | text_exts


def format_from_extension(
    filename: Union[Path, str]
):
    input_file=filename    
    ext=input_file.split('.')[-1]
    if ext.startswith('h5'):
        ext='h5'
    elif ext in avail_exts:
        ext=ext
    else:
        raise ValueError('Cannot determine the format from the file extension, please specify the format using the `input_format` parameter')
    return ext



def txt_format(
    filename: Union[Path, str],
    delimiter: Optional[str] = '\t',
    input_format: Optional[str] = None
):
    input_file=Path(filename)
    if input_format=='csv':
        df=pd.read_csv(input_file, nrows=5)
        cols=len(df.columns)
    else:
        df=pd.read_csv(input_file, nrows=5,delimiter=delimiter)
        cols=len(df.columns)
    if cols==1:
        warnings.warn("your table seems to have only 1 columns, you may have set the wrong delimiter. You can set the delimiter using the `delimiter` parameter")
    elif cols<5:
        return 'coo'  #tab-delimited sparse triplet file (bin1, bin2, count)
    else:
        return 'bg2'  #2D bedGraph-like file (chrom1, start1, end1, chrom2, start2, end2, count)


def to_cool(
    filename: Union[Path, str],
    bin_size: int,
    output_file= Union[Path, str],
    input_format: Optional[str] = None,
    delimiter: Optional[str] = '\t',
    chroms_file: Optional[Union[Path, str]]=None,
    hic_pro_bed: Optional[Union[Path, str]]=None
):
    input_file=filename    
    #1. input format determination
    if input_format==None:
        input_format=format_from_extension(input_file)
    else:
        input_format=input_format
    #2. File conversion
    input_file=Path(filename)      # allow passing strings
    ### 2D-text files as input
    if input_format in text_exts:
        if chroms_file==None:
                raise ValueError('in order to convert 2D-text file to cool, you need to provide the chromosome size file')
        args=['cooler', 'load', '-f', txt_format(input_file,delimiter,input_format), f'{chroms_file}:{bin_size}', input_file, output_file]
        p = subprocess.run(args)
    ### other formats
    elif input_format in avail_exts:
        if input_format=='hicpro':
            if hic_pro_bed==None:
                raise ValueError('in order to convert hicpro files to cool, you need to provide the hicpro-generated bed file')
            args=['hicConvertFormat', '-m', input_file, '--bedFileHicpro',hic_pro_bed, '--inputFormat', 'hicpro', '--outputFormat', 'cool', '-o', output_file]
            p = subprocess.run(args)
        else:
            args=['hicConvertFormat', '-m', input_file, '-o', output_file, '--resolutions', str(bin_size), '--inputFormat', input_format, '--outputFormat', 'cool']
            p = subprocess.run(args)
    else:
        raise ValueError('Invalid input format --> here you are the suitable formats! `csv`, `tsv`, `bed`, `bedpe`, `coo`, `ginteractions`, `txt`, `h5`, `hicpro`, `hic`, `cool`')
        
        
        