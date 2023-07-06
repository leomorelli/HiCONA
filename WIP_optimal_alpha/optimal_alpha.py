import hicona_cooler
import pandas as pd
import cooler
import numpy as np
import matplotlib.pyplot as plt
import math


def line_intersection(line1, line2):
    # Intersection of two lines (first one)
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception("lines do not intersect")
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def n_nodes_edges(cool_file, alpha_thr):
    # Compute node and edges given alpha threshold (rows are thresholds)
    DF = pd.DataFrame()
    c = cool_file
    big_net = pd.concat(grp[0] for grp in c.tables(count_thr=0, dist_thr=200000000))
    for a in alpha_thr:
        net_filt = big_net[big_net["spar_alpha"] <= a]
        n_edges = net_filt.shape[0]
        n_nodes = len(set(net_filt["bin1_id"]) | set(net_filt["bin1_id"]))
        df = pd.DataFrame(
            [a, n_nodes, n_edges], index=["alpha", "n_nodes", "n_edges"]
        ).T
        DF = pd.concat([DF, df])
    return DF


def local_alpha(DF):
    # Euclidean distance of each alpha threshold from intersection
    df_f = DF
    df_f.index = [x for x in range(df_f.shape[0])]
    df_f["edges_f"] = df_f["n_edges"].tolist() / df_f["n_edges"].max()
    df_f["nodes_f"] = df_f["n_nodes"].tolist() / df_f["n_nodes"].max()
    nodes = df_f["nodes_f"].tolist()
    edges = df_f["edges_f"].tolist()
    line1 = ([nodes[0], edges[0]], [nodes[-1], edges[0]])
    line2 = ([nodes[-1], edges[0]], [nodes[-1], edges[-1]])
    inter = line_intersection(line1, line2)
    dist = []
    for a in df_f["alpha"]:
        x = df_f[df_f["alpha"] == a]["n_nodes"] / df_f["n_nodes"].max()
        y = df_f[df_f["alpha"] == a]["n_edges"] / df_f["n_edges"].max()
        dist.append(math.dist(list(inter), [x, y]))
    alpha_thr = df_f.iloc[dist.index(min(dist)), :]["alpha"]
    return alpha_thr, df_f


def optimal_alpha(cool_file):
    # Iterate function to better approximate value
    c = cool_file
    i = 0
    thr_extremes = [1, 8]
    steps = [10, 100, 1000, 0]
    df = pd.DataFrame()
    while i < 3:
        c = cool_file
        alpha_thr = list(
            np.array([x for x in range(thr_extremes[0], thr_extremes[1])]) / steps[i]
        )
        alpha_thr = [0.0001] + alpha_thr + [0.9999]
        DF = n_nodes_edges(c, alpha_thr)
        alpha, DF_alpha = local_alpha(DF)
        df = pd.concat([df, DF_alpha])
        opt_a = alpha_thr.index(alpha)
        print(alpha)
        i = i + 1
        thr_extremes = [
            int(alpha_thr[opt_a - 1] * steps[i]),
            int(alpha_thr[opt_a + 1] * steps[i]),
        ]
    return alpha, df


# Plotting function?
# func -> give optimal or used defined alpha, create cool file filtered cause input is required
