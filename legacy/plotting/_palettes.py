import seaborn as sns
import matplotlib.colors as co

cluster_palette = co.ListedColormap(
    ["#FFFFFF"] + [co.to_hex(color) for color in sns.color_palette("tab20", 500)],
    name="cluster_palette",
)

prob_palette = sns.color_palette("coolwarm", as_cmap=True)
