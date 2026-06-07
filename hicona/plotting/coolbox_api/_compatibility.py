"""Stylesheet-compatible replacements for native coolbox tracks.

By default, many coolbox tracks do not work well with matplotlib style sheets
because their default parameters (e.g. ``ChromName`` property ``fontsize``) are
declared as class-level defaults that the style sheet cannot override.

The classes here inherit from their coolbox counterparts, removing or redeclaring
the offending properties so that style sheets take effect correctly.
"""

import coolbox.api as ca

__all__ = ("XAxis", "ChromName", "HicMatBase")


class XAxis(ca.XAxis):
    """Stylesheet-compatible genomic x-axis track."""

    DEFAULT_PROPERTIES = {
        "where": "bottom",
        "height": 1,
    }

    def plot(self, ax, gr: ca.GenomeRange, **kwargs):
        """Plot the x-axis with auto-scaled genomic coordinate labels."""
        self.ax = ax

        ax.set_xlim(gr.start, gr.end)
        ticks = ax.get_xticks()

        if ticks[-1] - ticks[1] <= 1000:
            labels = ["{:.0f}".format((x)) for x in ticks]
            labels[-2] += " bp"
        elif ticks[-1] - ticks[1] <= 1e5:
            labels = ["{:,.0f}".format((x / 1e3)) for x in ticks]
            labels[-2] += " Kb"
        elif 1e5 < ticks[-1] - ticks[1] < 4e6:
            labels = ["{:,.0f}".format((x / 1e3)) for x in ticks]
            labels[-2] += " Kb"
        else:
            labels = ["{:,.1f} ".format((x / 1e6)) for x in ticks]
            labels[-2] += " Mbp"

        ax.axis["x"] = ax.new_floating_axis(0, 0.5)
        ax.axis["x"].axis.set_ticklabels(labels)
        ax.axis["x"].axis.set_tick_params(which="minor", bottom="on")

        ax.axis["x"].major_ticklabels.set(size="small")

        if "where" in self.properties and self.properties["where"] == "top":
            ax.axis["x"].set_axis_direction("top")


class ChromName(ca.ChromName):
    """Stylesheet-compatible chromosome name label track."""

    DEFAULT_PROPERTIES = {"offset": 0.45}

    def plot(self, ax, gr: ca.GenomeRange, **kwargs):
        """Plot the chromosome name at the specified offset within the region."""
        x = gr.start + self.properties["offset"] * (gr.end - gr.start)  # type: ignore
        ax.text(x, 0, gr.chrom, size="x-large")
        ax.set_xlim(gr.start, gr.end)


class HicMatBase(ca.HicMatBase):
    """Stylesheet-compatible base class for Hi-C matrix tracks."""

    def plot_label(self):
        """Overriding inherited method to remove forcing text dimension."""

        if hasattr(self, "label_ax") and self.label_ax is not None:
            self.label_ax.text(
                0.15,
                0.5,
                self.properties["title"],
                horizontalalignment="left",
                verticalalignment="center",
            )
