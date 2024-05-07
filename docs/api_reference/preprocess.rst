.. _api.preprocess:

=============
Preprocessing
=============

.. currentmodule:: hicona.preprocess


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
