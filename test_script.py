import os

import pandas as pd

from hicona import HiconaGraph

FOLDERS = [
    # "test_files/Rao_2014_HUVEC_top_500000",
    # "test_files/Rao_2014_IMR90_top_500000",
    # "test_files/Rao_2014_HUVEC_full",
    "test_files/Rao_2014_IMR90_full",
]
PERMS = [["1_TssA", "Bound"], ["1_TssA"], ["7_Enh", "Bound"], ["7_Enh"]]
"""
PERMS = [
    ["H3K27ac"],
    ["H3K27me3"],
    ["H3K4me1"],
    ["H3K4me3"],
    ["H3K9me3"],
    ["H3K27ac", "H3K27me3"],
    ["H3K27ac", "H3K9me3"],
    ["H3K4me3", "H3K27me3"],
    ["H3K4me3", "H3K9me3"],
]
"""
NUM_PERMS = 1_000
STATS = ["betweenness", "ave_degree"]  # "betweenness"]
ANN_FILE = "bin_annotation.csv"
ALPHA_THR = 0.15
OUT_FOLDER = f"permutation_results_{ALPHA_THR}_OPT"

for folder in FOLDERS:
    for stat in STATS:
        fold_name = folder.split("/")[-1]
        print(f"Working on folder: {fold_name}")
        out_dir = os.path.join(OUT_FOLDER, f"plots_{fold_name}_{stat}")
        os.makedirs(out_dir)
        p_vals = []
        annot_df = pd.read_csv(os.path.join(folder, ANN_FILE), index_col=0)
        chroms = [f for f in os.listdir(folder) if f.startswith("chr")]
        for file in chroms:
            print(f"--Working on chromosome: {file}")
            chr_df = pd.read_csv(os.path.join(folder, file), index_col=0)
            chr_df = chr_df[chr_df["spar_alpha"] < ALPHA_THR]
            network = HiconaGraph(chr_df, ann_df=annot_df)
            for perm in PERMS:
                print(f"----Working on permutation: {perm}")
                col_a = perm[0]
                col_b = perm[1] if len(perm) == 2 else "universe"
                test = f"{col_a}_vs_{col_b}"
                plt_path = f"{file.split('.')[0]}_{stat}_{test}_{NUM_PERMS}.png"
                plt_path = os.path.join(out_dir, plt_path)
                perm_res = network.permute_attributes(
                    perm,
                    stat,
                    NUM_PERMS,
                    plot_path=plt_path,
                )
                print(f"----Got {perm_res}")
                perm_res = perm_res[0]
                perm_res["chr"] = file[3:-4]
                perm_res["a"] = col_a
                perm_res["b"] = col_b
                p_vals.append(perm_res)

        pvals_df = pd.DataFrame(p_vals)
        pvals_df.to_csv(os.path.join(out_dir, f"pvals_{stat}.csv"))
