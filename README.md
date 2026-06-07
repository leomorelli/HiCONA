# HiCONA - Hi-C Organization Network Analysis

## Motivation 
Any genomic contact map can be seen as an adjacency matrix, and thus a natural
representation for a graph. Despite this fact, contact maps are hardly ever 
analyzed as graphs. Probably the main reasons for this is the
difficulty in handling these huge sparse matrices.

**HiCONA** is a Python package whose aim is to provide the tools to perform
network analysis on genomic contact maps. HiCONA provides functionalities for
I/O, graph creation and manipulation, plotting and overall a customizable
interface which can be used with the provided methods or user-defined ones.

## Installation

HiCONA requires Python ≥3.10 and is supported on Linux and macOS only.
This is due to the required dependency [graph-tool](https://graph-tool.skewed.de/).
For the same reason, installation through conda-forge is recommended (the usage
of [mamba](https://github.com/mamba-org/mamba) instead of conda is strongly 
advised for faster dependency solving).

```sh
conda create -n hicona_env
conda activate hicona_env
conda install -c conda-forge hicona
```

Installation through pip is also possible, though an available installation of
graph-tool in the environment is required.

```sh
pip install hicona
```

Installation from source (mainly for development purposes) is possible using the
provided environment.yaml file.

```bash
git clone git@github.com:leomorelli/HiCONA.git hicona
cd hicona
conda create -f environment.yaml
conda activate hicona_env
```

## Documentation

API documentation and examples will soon be available on ReadTheDocs.

## Citing

Reference to the publication will be available soon. For the time being, if 
you find this package useful, please cite it by providing the link to 
[this page](https://github.com/leomorelli/HiCONA).

## Contributing

Contributing guidelines will be available soon.
