Installation
============

Due to one of the dependencies (:py:mod:`graph_tool`) not being installable through
``pip``, installing through ``conda``/``mamba`` is highly advised.

.. note::
   Using ``mamba`` should be preferred over ``conda``, as the latter might require hours to
   solve the environment. To use ``conda``, simply replace ``mamba`` with ``conda`` in
   all the following commands.

.. warning::
   When installing from source or PyPI, :py:mod:`graph_tool` **MUST** be installed before
   :py:mod:`hicona`. Failure to do so might result in package version mismatch.

Source
------

::

  $ mamba create -n hicona_env python=3.11
  $ mamba activate hicona_env
  $ mamba install graph-tool
  $ git clone git@github.com:leomorelli/HiCONA.git hicona
  $ cd hicona
  $ pip install .

PyPI
----

Requires a previous accessible installation of :py:mod:`graph-tool`

::

  $ pip install hicona


Conda/Mamba
-----------

::

  $ mamba create -n hicona_env python=3.11
  $ mamba activate hicona_env
  $ mamba install hicona
