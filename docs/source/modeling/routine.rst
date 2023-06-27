Routine
=======

Routine includes three levels definition, which are descriptive routine data
and model, optimization model, and data mapping.

Routine data and model
----------------------

Dispatch routine is the descriptive model of the optimization problem.

Further, to facilitate the routine definition, AMS developed a class
:py:mod:`ams.core.param.RParam` to pass the model data to multiple routine modeling.

.. autoclass:: ams.core.param.RParam
    :noindex:

.. currentmodule:: ams.routines
.. autosummary::
      :recursive:
      :toctree: _generated

      RoutineData
      RoutineModel

Optimization model
------------------

Optimization model is the optimization problem. ``Var``, ``Constraint``, and
``Objective`` are the basic building blocks of the optimization model. ``OModel``
is the container of the optimization model.
A summary table is shown below.

.. currentmodule:: ams.opt
.. autosummary::
      :recursive:
      :toctree: _generated

      Var
      Constraint
      Objective
      OModel

Data mapping
------------

Data mapping defines the relationship between AMS routine results and the
dynamic simulator ANDES. The dynamic module, :py:mod:`ams.interop.andes.Dynamic`,
is responsible for the conversion and synchronization of data between AMS and ANDES.

.. autoclass:: ams.interop.andes.Dynamic
    :noindex:
    :members: send, receive

When using this interface, the dynamic or static model is automatically selected
based on the initialization status of the TDS. For more detailed information about
the implementation of :py:mod:`ams.interop.andes.Dynamic.send` and
:py:mod:`ams.interop.andes.Dynamic.receive`, refer to the full API reference or
examine the source code.

.. note::
      Check ANDES documentation
      `StaticGen <https://docs.andes.app/en/latest/groupdoc/StaticGen.html#staticgen>`_
      for more details about substituting static generators with dynamic generators.