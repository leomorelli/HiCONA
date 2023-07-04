import sys

sys.path.insert(0, "..")
import hicona

handle = hicona.HiconaCooler(
    "Rao_2014_HUVEC_MboI_4DNFIRMZ7QTE.mcool::resolutions/10000"
)
print(handle)
