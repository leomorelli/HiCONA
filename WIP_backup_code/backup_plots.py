# THRESHOLD, NON CUMULATIVE VERSION
# def plot_annot_dynamics(
#     ann_dynamics: pd.DataFrame,
#     alpha_distr: pd.DataFrame,
#     ann_name: str,
#     sort_rows: bool = True,
#     img_path: str = None,
#     show: bool = False,
# ):
#     """Placeholder."""

#     def get_color_map():
#         """Placeholder"""
#         cols = ["mediumblue", "blue", "white", "red", "firebrick"]
#         vals = [0, 0.15, 0.5, 0.85, 1]
#         cmap = LinearSegmentedColormap.from_list("rg", list(zip(vals, cols)))
#         return cmap

#     def get_row_order(dataf):
#         """Placeholder"""
#         link = linkage(dataf, optimal_ordering=True)
#         dendro = dendrogram(link, no_plot=True)
#         return dendro["leaves"]

#     ann_cols = [f"{ann_name}1", f"{ann_name}2"]
#     table = ann_dynamics.copy()

#     bkg = table.groupby(ann_cols)["num_pixels"].sum().reset_index()
#     bkg["num_pixels"] = bkg["num_pixels"] / bkg["num_pixels"].sum()
#     bkg["annot"] = bkg.pop(ann_cols[0]) + "-" + bkg.pop(ann_cols[1])
#     bkg.set_index(["annot"], inplace=True)

#     table["annot"] = table.pop(ann_cols[0]) + "-" + table.pop(ann_cols[1])
#     table = table.set_index(["annot", "alpha"]).squeeze().unstack()
#     # table = table.drop(columns=[c for c in table.columns if c > max_alpha])

#     for col in table.columns:
#         table[col] = table[col] / table[col].sum()
#         table[col] = table[col].divide(bkg["num_pixels"], fill_value=0)
#     table = np.log2(table)

#     if sort_rows:
#         table = table.iloc[get_row_order(table)]

#     fig, axes = plt.subplots(2, 1, height_ratios=[1, 3], figsize=(15, 10))

#     sns.lineplot(alpha_distr, ax=axes[0][0])
#     axes[0][0].set(
#         xlabel="Alpha",
#         ylabel="Number of Pixels",
#         # xticks=[0.05 * i for i in range(0, round(max_alpha / 0.05) + 1)],
#         # xlim=(0, max_alpha),
#         ylim=(-max(alpha_distr) * 0.1, max(alpha_distr) * 1.1),
#     )
#     axes[0][0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
#     axes[0][0].xaxis.set_label_coords(-0.05, -0.04)

#     sns.heatmap(
#         table,
#         robust=True,
#         cmap=get_color_map(),
#         cbar_kws={"location": "bottom", "shrink": 0.5, "pad": 0.05},
#         ax=axes[1][0],
#         yticklabels=table.index,
#         center=0,
#         # vmax=1.5,
#         # vmin=-1.5,
#     )
#     axes[1][0].set(xlabel=None, ylabel=None, xticklabels=[])
#     axes[1][0].tick_params(bottom=False)

#     fig.tight_layout(pad=0)

#     if img_path:
#         plt.savefig(img_path)

#     if show:
#         plt.show()


# CUMULATIVE VERSION
# def plot_annot_dynamics(
#     ann_dynamics: pd.DataFrame,
#     alpha_distr: pd.DataFrame,
#     ann_name: str,
#     img_path: str = None,
#     show: bool = False,
# ):
#     """Placeholder."""

#     table = ann_dynamics.copy()

#     table["annot"] = table.pop("HMM_annot1") + "-" + table.pop("HMM_annot2")
#     table = table.set_index(["annot", "alpha"]).squeeze().unstack()

#     _, axes = plt.subplots(2, 1)

#     sns.heatmap(table, robust=True, cbar=False, ax=axes[0][0])
#     sns.lineplot(alpha_distr, ax=axes[1][0])

#     axes[1][0].set_xlim(0, 0.65)
#     plt.show()


# Plotting individual matrices for dynamics
# ann_columns = [f"{ann_name}1", f"{ann_name}2"]

# alpha_vals = ann_dynamics["alpha"].unique()
# alpha_vals[::-1].sort()
# num_cols = 2  # 1 if len(alpha_vals) == 1 else 2
# num_rows = int(round_half_up(len(alpha_vals) / 2))

# _, axes = plt.subplots(num_rows, num_cols)

# for ind, val in enumerate(alpha_vals):
#     print(axes)
#     print(ind)
#     print(axes[ind])
#     table = ann_dynamics.query(f"alpha == {val}").drop(columns=["alpha"])
#     table = table.set_index(ann_columns).squeeze().unstack().T
#     print(table)
#     sns.heatmap(table, annot=True, ax=axes[ind])
