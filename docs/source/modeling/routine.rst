Routine
===========

Routine refers to dispatch-level model, and it includes two sectinos, namely,
Data Section and Model Section.

Data Section
------------------

A simplified code snippet for RTED is shown below as an example.

.. code-block:: python

    class RTED:

        def __init__(self):
            ... ...
            self.R10 = RParam(info='10-min ramp rate',
                              name='R10', tex_name=r'R_{10}',
                              model='StaticGen', src='R10',
                              unit='p.u./h',)
            self.gs = ZonalSum(u=self.zg, zone='Region',
                               name='gs', tex_name=r'S_{g}',
                               info='Sum Gen vars vector in shape of zone',
                               no_parse=True, sparse=True)
            ... ...
            self.rbu = Constraint(name='rbu', type='eq',
                                  info='RegUp reserve balance',
                                  e_str = 'gs @ mul(ug, pru) - dud')
            ... ...

Routine Parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As discussed in previous section, actual data parameters are stored in the device-level models.
Thus, in routines, parameters are retrieved from target devices given the device
name and the parameter name.
In the example above, ``R10`` is a 10-min ramp rate parameter for the static generator.
The parameter is retrieved from the devices ``StaticGen`` with the parameter name ``R10``.

Service
^^^^^^^^^^^^

Services are developed to assit the formulations.
In the example above, ``ZonalSum`` is a service to sum the generator variables in a zone.
Later, in the constraint, ``gs`` is multiplied to the reserve variable ``pru``.

Model Section
-----------------

Descriptive Formulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Dispatch routine is the descriptive model of the optimization problem.

Further, to facilitate the routine definition, AMS developed a class
:py:mod:`ams.core.param.RParam` to pass the model data to multiple routine modeling.

.. autoclass:: ams.core.param.RParam
    :noindex:

.. currentmodule:: ams.routines
.. autosummary::
      :recursive:
      :toctree: _generated

      RoutineBase

Numerical Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Interoperation with ANDES
-----------------------------------

The interoperation with dynamic simulator invovles both file conversion and data exchange.
In AMS, the built-in interface with ANDES is implemented in :py:mod:`ams.interop.andes`.


File Format Converter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Power flow data is the bridge between dispatch study and dynamic study,
where it defines grid topology and power flow.
An AMS case can be converted to an ANDES case, with the option to supply additional dynamic
data.

.. autofunction:: ams.interop.andes.to_andes
    :noindex:


Data Exchange in Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To achieve dispatch-dynamic cosimulation, it requires bi-directional data exchange between
dispatch and dynamic study.
From the perspective of AMS, two functions, ``send`` and ``receive``, are developed.
The maping relationship for a specific routine is defined in the routine class as ``map1`` and
``map2``.
Additionally, a link table for the ANDES case is used for the controller connections.

Module :py:mod:`ams.interop.andes.Dynamic`, contains the necessary functions and classes for
file conversion and data exchange.

.. autoclass:: ams.interop.andes.Dynamic
    :noindex:
    :members: send, receive

When you use this interface, it automatically picks either the dynamic or static model based on the TDS initialization status.
If the TDS is running, it selects the dynamic model; otherwise, it goes for the static model.
For more details, check out the full API reference or take a look at the source code.

.. note::
      Check ANDES documentation
      `StaticGen <https://docs.andes.app/en/latest/groupdoc/StaticGen.html#staticgen>`_
      for more details about substituting static generators with dynamic generators.