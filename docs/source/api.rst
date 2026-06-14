API Reference
=============


Main Classes
------------

The classes of the objects you will be most often interfacing yourself with when using HiCONA.
You should get acquainted with these classes to use HiCONA.

.. autosummary::
    :toctree: api

    hicona.HiconaCooler
    hicona.PixelTable
    hicona.BinTable
    hicona.HiconaGraph

Plotting
--------

Various functions to plot pixel matrices, including their annotations and clustering.
These functions favor ease of use at the expense of flexibility. For highly customized
plots, use the tracks provided by :py:mod:`hicona.plotting.coolbox_api` to compose your own plots
using :py:mod:`coolbox`.

.. autosummary::
    :toctree: api

    hicona.plotting.plot_comparison
    hicona.plotting.plot_clustering
    hicona.plotting.plot_table


Miscellaneous
-------------
Various utility functions and objects. These functionalities might be unstable and
might be changed or removed in future versions. Think of this section as "under
evaluation".

.. autosummary::
    :toctree: api

    hicona.misc.expected_normalized_cooler
    hicona.misc.expected_normalized_pixels
    hicona.misc.get_expected_counts
