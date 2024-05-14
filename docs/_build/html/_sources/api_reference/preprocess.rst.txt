.. _api.preprocess:

=============
Preprocessing
=============

.. currentmodule:: hicona.preprocess

The preprocessing module provides a set of functionalities to customize the
default preprocessing from the package. It mainly provides a :class:`Flow` class
and a set of preprocessing functions.

The :class:`Flow` can be used to specify the filtering and normalization operations
to be applied to the data (in an order sensitive manner); it can be created from 
scratch or by modifying a default one. Moreover these objects can be saved and loaded
for a later use.

The package provides a set of filtering and normalization functions, namely those
that are more likely to be used. Nevertheless, any function can be used as long as 
it satisfies the following conditions:

- It must take an instance of the :class:`~hicona.HiconaTable` as its first argument.
- All other arguments must be JSON-types.
- It must return an iterable of ``pandas.DataFrame`` objects representing processed
  table chunks.


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
    Flow.as_json
    Flow.save_to_json

Operations
~~~~~~~~~~

.. autosummary::
    :toctree: api/

    Flow.ops_add
    Flow.ops_remove
    Flow.ops_reset
    Flow.ops_show
    Flow.get_partials
    Flow.source_add


Filtering functions
-------------------
.. autosummary::
    :toctree: api/
    
    filt_genomic_dist
    filt_column_quant
    filt_column_value
    filt_inter_chroms
    filt_self_looping

Normalization functions
-----------------------
.. autosummary::
    :toctree: api/

    norm_genomic_dist
    norm_none
    norm_binwise
