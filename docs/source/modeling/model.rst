Model
=========

This section introduces the modeling of power system devices. Here the term
``model`` refers to the descriptive model of a device, which is used to
hold the model-level data and variables.

AMS follows the model organization design of `ANDES`, where two classes
defined in ANDES, `ModelData` and `Model`, are used.

Parameters
------------
Parameter is an atom element in building a power system model. Most parameters
are read from an input file, and other parameters are calculated from the existing
parameters.

AMS leverages the parameter definition in ANDES, where four classes,
``DataParam``, ``IdxParam``, ``NumParam``, and ``ExtParam`` are used.
More details can be found in ANDES documentation
`Development - Parameters <https://docs.andes.app/en/latest/modeling/parameters.html>`_.

Variables
-----------

.. autoclass:: ams.core.var.Algeb
    :noindex:

      Algeb

ModelData and Model
------------------------------


Examples
------------

Following two examples show how to define a device model in AMS.
