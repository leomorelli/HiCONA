# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys


sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "hicona"
copyright = "2025, Leonardo Morelli, Stefano Cretti"
author = "Leonardo Morelli, Stefano Cretti"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


autosummary_generate = True  # Ensure stub .rst files are created
autodoc_default_options = {
    # "members": True,
    # "undoc-members": True,
    # "show-inheritance": True,
    # "exclude-members": "__init__",
}
autodoc_typehints = "none"
autodoc_mock_imports = ["graph_tool"]
# autoclass_content = "class"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_show_sphinx = False
html_show_sourcelink = False

nbsphinx_execute = "never"
