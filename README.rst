HiCONA - Hi-C Organization Network Analysis
+++++++++++++++++++++++++++++++++++++++++++

**HiCONA** is a Python3 package whose aim is to provide the tools to perform
network analysis on Hi-C contact matrices. A Hi-C contact matrix is in fact
an adjacency matrix, and thus a natural representation for a graph, but
despite this fact it is hardly ever used as such. This is mostly due to
how difficult it can be to handle these huge sparse matrices.

HiCONA tries to address these needs by providing functionalities for I/O,
graph creation and manipulation, plotting and other quality of life functionalities.

Installation
============

From source
-----------

Currently, HiCONA can only be installed from source. To do so, first create a new
conda environment; this is required for one of the dependencies (graph-tool). We
strongly encourage the use of mamba as solver, since conda can take hours solving
the environment.

.. code-block::

    mamba create -n hicona_env python=3.11
    mamba activate hicona_env

Then, install the required dependency graph-tool::

    mamba install graph-tool


**NOTE**: graph-tool **MUST** be installed first (due to numpy dependency conflicts)
Then, simply clone the repository and install using pip::

   git clone git@github.com:leomorelli/HiCONA.git hicona
   cd hicona
   pip install .


Documentation
=============

Currently, documentation is available only via local build::

    mamba create -n hicona_docs python=3.11
    mamba activate hicona_docs
    mamba install graph-tool
    cd hicona
    pip install .[docs]
    cd docs
    git submodule update --remote --merge
    make clean html

The documentation will then be in ``hicona/docs/build/html``.

Citing
======

*As of yet, there are no publications on HiCONA.*


.. _Cooler: https://github.com/open2c/cooler
.. _graph-tool: https://graph-tool.skewed.de/
