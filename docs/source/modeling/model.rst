Model
=========

This section introduces the modeling of power system devices.

AMS follows the model organization design of `ANDES`, where two classes
defined in ANDES, `ModelData` and `Model`, are used.

Parameters
------------
Parameter is an atom elements in building a power system model. Most parameters
are read from an input file, and other parameters are calculated from the existing
parameters.

AMS leverages the parameter definition in ANDES, where four classes,
``DataParam``, ``IdxParam``, ``NumParam``, and ``ExtParam`` are used.

More details can be found in ANDES documentation
[Development - Parameters](https://docs.andes.app/en/latest/modeling/parameters.html).

Further, to facilitate the routine definition, AMS developed a class
``RParam`` to pass the model data to multiple routine modeling.

.. currentmodule:: ams.core.param
.. autosummary::
      :recursive:
      :toctree: _generated

      RParam

Variables
-----------

.. currentmodule:: ams.core.var
.. autosummary::
      :recursive:
      :toctree: _generated

      Algeb

ModelData and Model
------------------------------
