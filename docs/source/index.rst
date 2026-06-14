.. Home

HiCONA - Hi-C Organization with Network Analysis
=================================================

Any genomic contact map can be seen as an adjacency matrix, and thus a natural
representation for a graph. Despite this fact, contact maps are hardly ever
analyzed as graphs, mainly due to the difficulty of handling these huge sparse
matrices.

**HiCONA** (Hi-C Organization with Network Analysis) is a Python package whose
aim is to provide the tools to perform network analysis on genomic contact maps.
It provides functionalities for I/O, graph creation and manipulation, plotting,
and a customizable interface which can be used with the provided methods or
user-defined ones.

The two core data primitives are **bins** — fixed-size genomic intervals that
partition the reference genome — and **pixels** — pairs of bins with a measured
contact frequency. HiCONA provides three classes to work with this data,
each optimized for a different stage of the analysis. See :doc:`implementation`
for a detailed description.

.. toctree::
   :maxdepth: 3
   :hidden:

   installation
   implementation
   api
   notebooks/index
