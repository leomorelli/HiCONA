"""Test file for debuggin purposes"""

from hicona import HiconaCooler

COOL_PATH = "test_files/small.mcool::resolutions/10000"
hc = HiconaCooler(COOL_PATH)
hc.available_annotations()
