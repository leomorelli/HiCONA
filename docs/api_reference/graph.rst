.. _api.graph:

===========
HiconaGraph
===========

.. currentmodule:: hicona


The ``HiconaGraph`` class is used for the bulk of the network analysis in Hicona.
It allows to represent a processed (and usually sliced) part of the data as a graph,
with nodes representing bins (genomic regions) and edges representing pixels 
(interactions among them).

Since ``HiconaGraph`` extends the :class:`graph_tool.Graph` class, this means that:

- All sorts of bin and pixel annotations can be easily added as edge and node properties.
- All methods and attributes of the :class:`graph_tool.Graph` class are still available.
- Specialized versions of some network analysis algorithms are available.

.. note::
    All algorithms implemented by ``HiconaGraph`` are explained in depth in the 
    <add docs page> section of the documentation, while their usage is shown
    <add hyperlink>.

Constructor
-----------
.. autosummary::
    :toctree: api/

    HiconaGraph

Attributes
----------
.. autosummary::
    :toctree: api/

    HiconaGraph.table
    HiconaGraph.ids_table

Manipulation
------------
.. autosummary::
    :toctree: api/

    HiconaGraph.add_genomic_edges

Analysis
--------
.. autosummary::
    :toctree: api/

    HiconaGraph.compute_clustering
    HiconaGraph.permute_annotations
    