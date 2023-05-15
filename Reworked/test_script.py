"""Test file for debuggin purposes"""

from hicona import HiconaCooler

COOL_PATH = "test_files/copy_lieberman.mcool::resolutions/10000"
CHROM = "chr17"
hc = HiconaCooler(COOL_PATH)
hc.create_tables(dist_thr=200_000_000)
# for table in hc.tables([CHROM]):
#     print(table)
