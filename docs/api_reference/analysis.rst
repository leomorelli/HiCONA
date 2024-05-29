.. _api.analysis:

========
Analysis
========

.. currentmodule:: hicona.analysis

The analysis subpackage provides tools to perform, inspect and plot all sorts
of analyses on the data.
Though these objects can be initialized directly, in general they will be 
returned by some method of the analyzed object (e.i. 
:class:`~hicona.analysis.AlphaGrid` is returned by the
:meth:`~hicona.HiconaTable.get_alpha_grid` method of the ``HiconaTable`` class)
or by a factory function which takes as input more than one object (<this will
be the method for comparison functions which will be implemented>).

Alpha filtering
---------------
.. autosummary::
    :toctree: api/
    
    AlphaGrid
    AlphaGrid.optimal_alpha
    AlphaGrid.plot

Annotation dynamics
-------------------
.. autosummary::
    :toctree: api/
    
    AnnotDynamics
    AnnotDynamics.get_dynamics
    AnnotDynamics.plot_interval
    AnnotDynamics.plot_full

    