HiCONA - Hi-C Organization Network Analysis
===========================================

**HiCONA** is a Python3 package whose aim is to provide the tools to perform
network analysis on Hi-C contact matrices. A Hi-C contact matrix is in fact
an adjacency matrix, and thus a natural representation for a graph, but
despite this fact it is hardly ever used as such. This is mostly due to
how difficult it can be to handle these huge sparse matrices.

HiCONA tries to address these needs by building on top of two main packages:

* `Cooler`_ for I/O and storage related tasks
* `graph-tool`_ for graph processing and plotting

HiCONA adheres as much as possible to the cooler format specifications,
striving to maintain full compatibility and the ability to be integrated with
other tools.

Moreover HiCONA implements other algorithms, especially for preprocessing,
optimized for both computational and memory efficiency.

Installation
------------

From source
-----------

Currently, HiCONA can only be installed from source. To do so, first create a new
conda environment; this is required for one of the dependencies (graph-tool). We
strongly encourage the use of mamba as solver, since conda can take hours solving
the environment.

```
mamba create -n hicona_env python=3.11
mamba activate hicona_env
```

Install the required dependency graph-tool.

```
mamba install graph-tool
```

**NOTE**: graph-tool **MUST** be installed first (due to numpy dependency conflicts)
Then, simply clone the repository and install using pip

```
git clone git@github.com:leomorelli/HiCONA.git hicona
cd hicona
pip install .
```


Citing
------
*As of yet, there are no publications on HiCONA.*


.. _Cooler: https://github.com/open2c/cooler
.. _graph-tool: https://graph-tool.skewed.de/
