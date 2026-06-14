Installation
============

HiCONA requires Python ≥3.10 and is supported on Linux and macOS only,
due to the required dependency :py:mod:`graph_tool`.
Installation through conda-forge is recommended.

.. note::
   Using ``mamba`` instead of ``conda`` is strongly advised for faster
   dependency solving. To use ``conda``, simply replace ``mamba`` with
   ``conda`` in all the following commands.

Conda/Mamba (recommended)
--------------------------

::

  $ mamba create -n hicona_env
  $ mamba activate hicona_env
  $ mamba install -c conda-forge hicona

PyPI
----

Requires a pre-existing installation of :py:mod:`graph_tool` in the environment.

::

  $ pip install hicona

Source
------

For development purposes, use the provided ``environment.yaml`` file.

::

  $ git clone git@github.com:leomorelli/HiCONA.git hicona
  $ cd hicona
  $ mamba env create -f environment.yaml
  $ mamba activate hicona_env
