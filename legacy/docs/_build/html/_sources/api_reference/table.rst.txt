.. _api.table:

===========
HiconaTable
===========

.. currentmodule:: hicona

One of the main premises of this package is the fact that the full ``pixel``
table contained in a cooler file is generally too large to be loaded into
memory and analyzed (plus, it can be noisy or have other problems). To solve 
this issue, ``hicona`` allows to preprocess the full ``pixel`` table (see 
<add tutorial>) to obtain subsets which can be loaded as ``HiconaTable``
instances.

``HiconaTable`` objects can be then used to further subset and filter the
data, perform analyses and generate networks (instances of the :doc:`graph`
class).

Constructor
-----------
.. autosummary::
    :toctree: api/

    HiconaTable

Attributes
----------
.. autosummary::
    :toctree: api/

    HiconaTable.bin_size
    HiconaTable.chunk_size
    HiconaTable.flow
    HiconaTable.region
    HiconaTable.size
    HiconaTable.uris

Data
----
.. autosummary::
    :toctree: api/

    HiconaTable.chunks
    HiconaTable.dataframe

Interval
--------
.. autosummary::
    :toctree: api/

    HiconaTable.subset
    HiconaTable.reset_index

Analysis
--------
.. autosummary::
    :toctree: api/

    HiconaTable.get_distribution
    HiconaTable.get_alpha_grid
    HiconaTable.get_annot_dynamics
   
   
