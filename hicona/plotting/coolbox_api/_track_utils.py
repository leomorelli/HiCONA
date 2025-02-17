"""Code to handle track default parameters selection."""

from matplotlib.axes import Axes

__all__ = ("get_label_axis",)


def get_label_axis(plot_axes: Axes) -> Axes:
    """Get the axes in which to display plot label.

    This is a workaround to avoid overriding a huge part of coolbox just to get access
    to the grid object itself in a nicer way. Simply assume that the provided set of
    axes comes from the middle column of a frame, therefore there should always be a
    right column from which to fetch the axes in which to put the title.
    """

    figure = plot_axes.figure
    assert figure is not None
    plot_spec = plot_axes.get_subplotspec()
    assert plot_spec is not None
    grid_spec = plot_spec.get_gridspec()

    lab_spec = grid_spec[plot_spec.rowspan.start, plot_spec.colspan.start + 1]
    for candidate_ax in figure.axes:
        if candidate_ax.get_subplotspec() == lab_spec:
            return candidate_ax

    raise ValueError("No axis found to the immediate right.")
