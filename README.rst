HiCONA - Hi-C Organization Network Analysis
===========================================

About
-----
**HiCONA** is a Python3 package whose aim is to provide the tools to perform 
network analysis on Hi-C contact matrices. A Hi-C contact matrix is in fact
an adjacency matrix, and thus a natural representation for a graph, but 
despite this fact it is hardly ever used as such. This is mostly due to 
how difficult it can be to handle these huge sparse matrices. 

HiCONA tries to address these needs by building on top of two main packages:
- `Cooler`_ for I/O and storage related tasks
- `graph_tool`_ for graph processing and plotting
HiCONA adheres as much as possible to the cooler format specifications,
striving to maintain full compatibility and the ability to be integrated with
other tools.

Moreover HiCONA implements other algorithms, especially for preprocessing, 
optimized for both computational and memory efficiency.


Installation
------------
*TODO: Update installation guide, this is just a sketch*

Due to the usage of graph-tool as a dependency, the installation of HiCONA in
a dedicated environment is highly incentivised. Via Conda, create a new 
environment by running:

    conda create -n HiCONA

Currently, due to some compatibility issues, it is required to manually
install Cooler and graph-tool **in this order** (despite graph-tool
documentation suggesting to install graph-tool first):

    conda install -c conda-forge -c bioconda cooler
    conda install -c conda-forge graph-tool

Then one can finally install HiCONA:

    conda install -c conda-forge hicona

In the future the whole procedure will be streamlined.


Citing
------
As of yet, there are no publications on HiCONA.


.. _Cooler: https://github.com/open2c/cooler
.. _graph-tool: https://graph-tool.skewed.de/
