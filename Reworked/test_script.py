"""Test file for debuggin purposes"""

from hicona import HiconaCooler
import pandas as pd

COOL_PATH = "test_files/ann_test.mcool::resolutions/10000"
CHROM = "chr1"
hc = HiconaCooler(COOL_PATH)
# hc.create_tables(dist_thr=200_000_000)
for table in hc.tables([CHROM]):
    print(table)
