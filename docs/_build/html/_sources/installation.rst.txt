Installation
============
..
  TODO: Update, this is just a sketch

PyPI
----

*Currently installation from PyPI is not supported.*


Conda
-----

*This does not actually work, it is placeholder text.*

Due to the usage of graph-tool as a dependency, the installation of HiCONA in
a dedicated environment is highly incentivised. Via Conda, create a new 
environment, and activate it, by running:

.. code-block::

    conda create -n HiCONA
    conda activate HiCONA

Currently, due to some compatibility issues, it is required to manually
install Cooler and graph-tool **in this order** (despite graph-tool
documentation suggesting to install graph-tool first):

.. code-block::

    conda install -c conda-forge -c bioconda cooler
    conda install -c conda-forge graph-tool

Then one can finally install HiCONA:

.. code-block::

    conda install -c conda-forge hicona

In the future the whole procedure will be streamlined.
