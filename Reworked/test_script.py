"""Test file for debuggin purposes"""

from hicona import HiconaCooler

COOL_PATH = "test_files/small.mcool::resolutions/10000"
hc = HiconaCooler(COOL_PATH)
hc.available_annotations(show=True)
hc.add_bin_annotation(
    "test_files/promoter_with_info.bed",
    to_keep_cols=[None, "strand"],
)
print(hc.bins()[1:10])
hc.encode_annotation(["strand"])
print(hc.bins()[1:10])
