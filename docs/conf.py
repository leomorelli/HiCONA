# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "HiCONA"
copyright = "2024, Leonardo Morelli, Stefano Cretti"
author = "Leonardo Morelli, Stefano Cretti"

import sys
import os

sys.path.insert(0, os.path.abspath(".."))

import hicona

version = str(hicona.__version__)
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
]

templates_path = ["_templates"]
exclude_patterns = []

autodoc_typehints = "none"


# -- numpydoc configuration --------------------------------------------------
numpydoc_show_class_members = False
autoclass_content = "class"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# Configuration of sphinx.ext.coverage
coverage_show_missing_items = True

# Configuration of sphinx.ext.intersphinx

intersphinx_mapping = {
    "cooler": ("https://cooler.readthedocs.io/en/latest/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "graph_tool": ("https://graph-tool.skewed.de/static/doc/", None),
}
