.. _api.cooler:

============
HiconaCooler
============

.. currentmodule:: hicona

    
The ``HiconaCooler`` class is responsible for interfacing with cooler-like
files. This means that it is responsible for I/O operations to and from cooler
files, managing bin annotations as well as creating and retrieving processed 
pixel tables. Ideally one should use the ``HiconaCooler`` class to generate
the pixel tables (instances of the :doc:`table` class) of interest, and then
follow up with their analysis.

.. TODO: Add link to a tutorial on how to use the HiconaCooler class.

The ``HiconaCooler`` class is a strict subclass of the :class:`cooler.Cooler`
class, extending all its functionalities without modifying them. For this
reason, an object of the ``HiconaCooler`` class can always be used as a
``cooler.Cooler`` object. Since no functionality is overwritten, attributes 
and methods from that class are not documented here (though they are listed in
the :class:`~hicona.HiconaCooler` class constructor page); for those, please 
refer to the `cooler documentation <https://cooler.readthedocs.io/en/latest/>`_.


.. warning::
   
    ``HiconaCooler`` should not break any of the functionalities of the
    ``cooler.Cooler`` class, though this is not guaranteed. This is because
    testing at every update would be extremely time-consuming, and thus
    impractical. If you find any issues, please report them on
    `Github <https://github.com/leomorelli/HiCONA>`_.


Constructor
-----------
.. autosummary::
    :toctree: api/

    HiconaCooler

Attributes
----------
.. autosummary::
    :toctree: api/
    
    HiconaCooler.bare_bins
    HiconaCooler.tables_root

Tables
------
.. autosummary::
    :toctree: api/

    HiconaCooler.create_table
    HiconaCooler.fetch_table
    HiconaCooler.list_tables

Annotation
----------
.. autosummary::
    :toctree: api/

    HiconaCooler.annot_list
    HiconaCooler.add_bin_annot
    HiconaCooler.del_bin_annot
    HiconaCooler.hmm_bin_annot
    HiconaCooler.ohe_bin_annot
   