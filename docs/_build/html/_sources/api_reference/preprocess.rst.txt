.. _api.preprocess:

=============
Preprocessing
=============

.. currentmodule:: hicona.preprocess

The preprocessing module provides a set of functionalities to customize the
default preprocessing from the package. It mainly provides a :class:`Flow` class
and a set of preprocessing operations which can be added to it.

The :class:`Flow` class can be used to specify the filtering and normalization operations
to be applied to the data (in an order sensitive manner); it can be created from 
scratch or by modifying a default one. Moreover these objects can be saved and loaded
for a later use.

The package provides some filtering and normalization operations, namely those that are 
more likely to be used. Support for custom operation is currently being worked on.


Flow object
-----------

.. autosummary::
    :toctree: api/

    Flow

I/O methods
~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    Flow.from_default
    Flow.from_file
    Flow.from_json
    Flow.rename
    Flow.to_file
    Flow.to_json

Operations
~~~~~~~~~~

.. autosummary::
    :toctree: api/

    Flow.ops_add
    Flow.ops_reset

Filtering operations
--------------------
.. autosummary::
    :toctree: api/
    
    FiltGenomicDist
    FiltColumnQuant
    FiltColumnValue
    FiltInterChroms
    FiltSelfLooping

Normalization operations
------------------------
.. autosummary::
    :toctree: api/

    NormBinwise
    NormGenomicDist
