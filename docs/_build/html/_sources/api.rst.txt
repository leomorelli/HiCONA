API
===

Quick Reference
---------------

HiconaCooler - I/O and preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    hicona.HiconaCooler

Create and access chromosome-level tables.

.. autosummary::
    hicona.HiconaCooler.list_tables
    hicona.HiconaCooler.create_tables
    hicona.HiconaCooler.tables

Manipulate bin annotation.

.. autosummary::
    hicona.HiconaCooler.list_annotations
    hicona.HiconaCooler.add_bin_annotation
    hicona.HiconaCooler.del_bin_annotation
    hicona.HiconaCooler.annotation_to_ohe

File creation.

.. autosummary::
    hicona.HiconaCooler.gen_sparsified_cooler

HiconaGraph - Network analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    hicona.HiconaGraph

Network analysis algorithms.

.. autosummary::
    hicona.HiconaGraph.permute_annotations

----

HiconaCooler
------------

.. autoclass:: hicona.HiconaCooler
    :members:

----

HiconaGraph
-----------

.. autoclass:: hicona.HiconaGraph
    :members:
