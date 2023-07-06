"""Script to try out functions during development"""

import csv
from datetime import datetime

from hicona import HiconaCooler, HiconaGraph
from hicona.utils import annotation_combinations

IN_FOLDER = "test_files"
OUT_FOLDER = "results"
BIN_SIZES = [10_000]  # 5_000]

"""
FILES = [
    {
        "name": "Rao_2014_HUVEC_MboI_4DNFIRMZ7QTE.mcool",
        "alpha_threshold": 0.21,
    },
    {
        "name": "Rao_2014_HMEC_MboI_4DNFIGUIV5KO.mcool",
        "alpha_threshold": 0.2,
    },
    {
        "name": "Rao_2014_IMR90_MboI_4DNFIJTOIGOI.mcool",
        "alpha_threshold": 0.16,
    },
    {
        "name": "Rao_2014_GM_MboI_4DNFIXP4QG5B.mcool",
        "alpha_threshold": 0.15,
    },
]
"""

FILES = [
    {
        "name": "Rao_2014_HUVEC_MboI_4DNFIRMZ7QTE.mcool",
        "alpha_threshold": 0.05,
    },
    {
        "name": "Rao_2014_HUVEC_MboI_4DNFIRMZ7QTE.mcool",
        "alpha_threshold": 0.10,
    },
    {
        "name": "Rao_2014_HUVEC_MboI_4DNFIRMZ7QTE.mcool",
        "alpha_threshold": 0.15,
    },
    {
        "name": "Rao_2014_HUVEC_MboI_4DNFIRMZ7QTE.mcool",
        "alpha_threshold": 0.20,
    },
    {
        "name": "Rao_2014_HUVEC_MboI_4DNFIRMZ7QTE.mcool",
        "alpha_threshold": 0.25,
    },
    {
        "name": "Rao_2014_HUVEC_MboI_4DNFIRMZ7QTE.mcool",
        "alpha_threshold": 0.30,
    },
    {
        "name": "Rao_2014_HUVEC_MboI_4DNFIRMZ7QTE.mcool",
        "alpha_threshold": 0.35,
    },
    {
        "name": "Rao_2014_HUVEC_MboI_4DNFIRMZ7QTE.mcool",
        "alpha_threshold": 0.40,
    },
]

ANNOTS = [
    "heterochromatin",
    "enhancer",
    "polycomb",
    "promoter",
]  # "Bound"]
STAT = "ave_degree"  # "ave_degree"
NUM_PERMS = 1_000

# Create output file
res_path = f"permutation_results_{datetime.now().strftime('%Y%m%d_%H:%M:%S')}"
res_path = f"{OUT_FOLDER}/{res_path}"
with open(res_path, "w", encoding="utf-8") as out_file:
    writer = csv.writer(out_file)
    writer.writerow(
        [
            "file_name",
            "bin_size",
            "chrom",
            "alpha_threshold",
            "a",
            "b",
            "pval",
            "num_perms",
            "statistic",
        ]
    )

# Pocess the files
for bin_size in BIN_SIZES:
    print(f"Working of statistic: {STAT}")
    print(f"Working of bin size: {bin_size}")

    for file_dict in FILES:
        print(f"-- Working on file: {file_dict['name']}")
        print(f"-- Working with alpha_threshold: {file_dict['alpha_threshold']}")

        file_path = f"{IN_FOLDER}/{file_dict['name']}::resolutions/{bin_size}"
        handle = HiconaCooler(file_path)
        ann_df = handle.bins()[:][ANNOTS]
        tables_iterator = handle.tables(dist_thr=200_000_000)
        assert len(tables_iterator) == 24

        for table, info in tables_iterator:
            print(f"---- Working on table: {info['chromosome']}")

            table = table[table["spar_alpha"] < file_dict["alpha_threshold"]]
            chrom_graph = HiconaGraph(table, ann_df=ann_df)
            perm_res = chrom_graph.permute_annotations(ANNOTS, STAT, NUM_PERMS)

            with open(res_path, "a", encoding="utf-8") as out_file:
                writer = csv.writer(out_file)
                for res in perm_res:
                    writer.writerow(
                        [
                            file_dict["name"],
                            bin_size,
                            info["chromosome"],
                            file_dict["alpha_threshold"],
                            res["a"],
                            res["b"],
                            res["pval"],
                            res["num_perms"],
                            res["statistic"],
                        ]
                    )
