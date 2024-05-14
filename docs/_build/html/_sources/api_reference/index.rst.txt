.. _api:

=============
API reference
=============

HiCONA revolves around three main classes, those being :doc:`cooler`,
:doc:`table` and :doc:`table`. These objects are directly available
in the ``hicona.*`` namespace.

Moreover, some public packages are available to extend the functionalities of
the base objects. Though these packages will rarely be accessed directly, especially the 
class constructors, they are still made public for documentation purposes. 
Currently, the available packages are:

- ``hicona.analysis``: Functions and classes to analyze the processed pixel tables.
- ``hicona.proprocess``: Function and classes to customize the pixel table creation process.


Quick reference list
~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2
    
    cooler
    table
    graph 
    preprocess
    analysis
