"""Placeholder"""


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


# TODO: check that the table is normalized.
