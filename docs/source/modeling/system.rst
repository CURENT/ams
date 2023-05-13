System
======

Overview
--------
System is the top-level class for organizing power system dispatch models
and routines. The full API reference of System is found at
:py:mod:`ams.system.System`.

Dynamic Imports
```````````````
System dynamically imports groups, models, and routines at creation. To add new
models, groups or routines, edit the corresponding file by adding entries
following examples.

.. autofunction:: ams.system.System.import_models
    :noindex:

.. autofunction:: ams.system.System.import_groups
    :noindex:

.. autofunction:: ams.system.System.import_routines
    :noindex:

Optimization Formulations
--------------------------------------------

In AMS, the dispatch modeling is formulated as standard optimization problem.

.. autofunction:: ams.opt.omodel.OModel
    :noindex:

