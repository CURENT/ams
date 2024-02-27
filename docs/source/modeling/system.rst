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

.. autofunction:: ams.system.System.import_types
    :noindex:


Device-level Models
------------------------
AMS follows a similar device-level model organization of ANDES with a few differences.


Routine-level Models
------------------------
In AMS, routines are responsible for collecting data, defining optimization
problems, and solving them.


Optimization
--------------------------------------------

Within the ``Routine``, the descriptive formulation are translated into
`CVXPY <https://www.cvxpy.org/>`_ optimization problem with
``Vars``, ``Constraints``, and ``Objective``.
The full API reference of them can be found in :py:mod:`ams.opt.Var`,
:py:mod:`ams.opt.Constraint`, and :py:mod:`ams.opt.Objective`.

.. autoclass:: ams.opt.omodel.OModel
    :noindex:
