"""Test file for debuggin purposes"""

from hicona_chunked import HiconaCooler

COOL_PATH = "test_files/small.mcool::resolutions/10000"
hc = HiconaCooler(COOL_PATH)
# hc.create_tables(chrom_selection=["chr2"], dist_thr=200_000_000)
hc.create_tables(chrom_selection=["chr6"], dist_thr=200_000_000)
# hc.create_tables(chrom_selection=["chr2"], dist_thr=2_000_000)
