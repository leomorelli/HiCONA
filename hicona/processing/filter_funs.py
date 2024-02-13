"""Placeholder"""

# TODO: check whether the table is normalized.


def max_genomic_dist(table, max_dist):
    """
    Remove pixels whose genomic distance is higher than a threshold.
    Params:
        - max_dist: max allowed genomic distance among bins in bp.
    """

    for chunk in table.chunks():
        dist = (chunk.bin2_id - chunk.bin1_id) * table.binsize
        chunk = chunk.loc[dist <= max_dist]
        yield chunk


def norm_count_quant(table, quant):
    """
    Remove pixels whose normalized counts value is below a threshold.
    Params:
        - quant: remove pixels whose normalized value is below this quantile.
    """

    quant_val = quant  # TODO: Compute table n-th quantile
    for chunk in table.chunks():
        chunk = chunk.loc[chunk.norm > quant_val]
        yield chunk


def self_looping_pix(table):
    """
    Remove self-looping pixels (pixels where bin1_id == bin2_id).
    Params:
        - None
    """

    for chunk in table.chunks():
        chunk = chunk.loc[chunk.bin1_id != chunk.bin2_id]
        yield chunk


def min_raw_counts(table, min_val):
    """
    Remove pixels whose raw count value is not greater than a threshold.
    Params:
        - min_val: remove pixels whose count is not greater than this value.
    """
    
    for chunk in table.chunks():
    chunk = chunk.loc[chunk.count > min_val]
        yield chunk

def rm_inter_chroms(table):
    """
    Remove inter chromosomal pixels.
    """
    pass
