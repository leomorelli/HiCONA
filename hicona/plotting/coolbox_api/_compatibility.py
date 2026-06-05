"""Alternative versions of tracks present natively in coolbox for plotting compatibility.

By default, many tracks provided by coolbox do not work well with a style sheet.
This is because some of the default parameters are simply aliases for matplotlib
properties (e.i. ChromName property `fontsize`). Since these properties have a
declared default value inside the class declaration, the style sheet will never
be able to override the default values. Other times, there is a value directly
inside the declaration of the plotting function.

There is probably a smarted and more elegant way of doing it, but a quick way
to allow style sheets to be used is to simply inherit from the classes redeclaring
or removing the property. For this reason this script is mostly a copy paste from
coolbox source code. Hopefully this behavior is changed in the future.
"""

import coolbox.api as ca

__all__ = ("XAxis", "ChromName", "HicMatBase")


class XAxis(ca.XAxis):
    DEFAULT_PROPERTIES = {
        "where": "bottom",
        "height": 1,
    }

    def plot(self, ax, gr: ca.GenomeRange, **kwargs):
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
    DEFAULT_PROPERTIES = {"offset": 0.45}

    def plot(self, ax, gr: ca.GenomeRange, **kwargs):
        x = gr.start + self.properties["offset"] * (gr.end - gr.start)  # type: ignore
        ax.text(x, 0, gr.chrom, size="x-large")
        ax.set_xlim(gr.start, gr.end)


class HicMatBase(ca.HicMatBase):
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
