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


Models
-----------
AMS follows the model organization design of ANDES.


Routines
-----------
In AMS, routines are responsible for collecting data, defining optimization
problems, and solving them.


Optimization
--------------------------------------------

In AMS, the dispatch is formulated as `CVXPY <https://www.cvxpy.org/>`_
optimization problem with ``Vars``, ``Constraints``, and ``Objective``.
The full API reference of them can be found in :py:mod:`ams.opt.Var`,
:py:mod:`ams.opt.Constraint`, and :py:mod:`ams.opt.Objective`.

.. autoclass:: ams.opt.omodel.OModel
    :noindex:
