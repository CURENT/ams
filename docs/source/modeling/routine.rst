Routine
===========

Routine refers to scheduling-level model, and it includes two sectinos, namely,
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
            self.gs = ZonalSum(u=self.zg, zone='Zone',
                               name='gs', tex_name=r'S_{g}',
                               info='Sum Gen vars vector in shape of zone',
                               no_parse=True, sparse=True)
            ... ...
            self.rbu = Constraint(name='rbu', is_eq=True,
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
Scheduling routine is the descriptive model of the optimization problem.

Further, to facilitate the routine definition, AMS developed a class
:py:mod:`ams.core.param.RParam` to pass the model data to multiple routine modeling.

.. autoclass:: ams.core.param.RParam
    :noindex:

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

Expression Notation in ``e_str``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Routine constraints, objectives, and expressions are written as Python source
strings in the ``e_str`` argument. Before being passed to CVXPY, ``e_str`` is
rewritten by :py:class:`ams.core.symprocessor.SymProcessor` so that bare names
like ``pg`` or ``Cft`` resolve to the underlying CVXPY ``Variable`` /
``Parameter`` / sparse-matrix objects of the host routine.

Multiplication is the easiest place to introduce silent bugs because there are
**three semantically distinct operations** that all read like "multiply":

==================  ==========================  ==============================================
Notation in e_str   Meaning                     When to use
==================  ==========================  ==============================================
``a @ b``           Matrix / matrix-vector mul  Topology matrix times a stacked vector, e.g.
                                                ``Cft @ pl``, ``PTDF @ p``.
``mul(a, b)``       Element-wise multiply       Per-device scaling, e.g. ``mul(ug, pg)`` to
                    (alias for ``cp.multiply``) gate generator output by its commitment.
                                                Broadcasts a scalar / 1-D parameter against a
                                                vector variable.
``2 * x``,          Scalar multiply             A literal number or a 0-D parameter scaling an
``coeff * y``                                   expression. CVXPY accepts ``*`` here.
==================  ==========================  ==============================================

Two convenience aliases exist for readability:

* ``a dot b`` — rewritten to ``a * b`` (kept for legacy expressions; typically
  used for scalar-times-expression forms such as ``t dot expr``). Note this
  is *not* an alias for ``mul()`` / element-wise multiply.
* ``multiply(a, b)`` — same as ``mul(a, b)``.

**Bare ``a * b`` between two identifiers is not rewritten and will be passed
to CVXPY as-is.** CVXPY will then either raise a shape error or, worse,
emit a deprecation warning and silently perform matrix multiplication. Always
write ``@`` or ``mul()`` explicitly. As of refactor step 1.4 the
historical catch-all rewrite ``word * word`` → ``word @ word`` has been
removed; existing routines were audited and the only affected expression
(``dopf.lvd``) was migrated to ``mul()``.

Other ``e_str`` rewrites worth knowing:

* ``sum(x)``, ``norm(x)``, ``pos(x)``, ``square(x)``, ``power(x, n)``,
  ``vstack(...)``, ``maximum(...)``, ``minimum(...)``, ``quad_form(...)``,
  ``sum_squares(...)``, ``diag(...)`` are forwarded to their ``cp.*``
  equivalents.
* Comparisons ``... == 0`` and ``... <= 0`` define equality and inequality
  constraints respectively (set via the ``is_eq`` flag on ``Constraint``).
* Powers use ``**`` (e.g. ``vmax**2``).

If in doubt, check an existing routine such as :py:mod:`ams.routines.dcopf`
or :py:mod:`ams.routines.rted` for canonical patterns.

How ``e_str`` becomes a CVXPY problem (codegen)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``e_str`` rewrites above are applied **once at prep time**, not on
every routine init. The first time a routine is instantiated, AMS:

1. Walks the constructed routine's ``constrs`` / ``exprs`` / ``exprcs``
   / ``obj`` registries.
2. Applies the ``sub_map`` rewrites to each ``e_str``.
3. Emits a small Python module at ``~/.ams/pycode/<routine>.py`` with
   one named callable per opt element — e.g.
   ``def _constr_pglb(r): return -r.pg + r.pmine`` — plus the
   pre-rendered LaTeX string for documentation.
4. Wires those callables onto the routine's ``Constraint`` /
   ``Expression`` / ``Objective`` instances via their ``e_fn``
   attribute.

Subsequent inits skip step 2-3 and just import. The cache is keyed by
md5 of the routine source file; editing an ``e_str`` regenerates
automatically. The whole cache can be refreshed or wiped via
``ams prep`` (see :ref:`ReleaseNotes` v1.2.2).

This is analogous to ANDES's ``andes prep`` / ``~/.andes/pycode/``
pipeline. **Author-facing API is unchanged** — you keep writing
``e_str``. The codegen is what runs underneath.

Authors who prefer to skip the DSL and write a callable directly may
pass ``e_fn=callable`` instead of ``e_str``; the runtime accepts both,
and the codegen leaves manually-set ``e_fn`` alone.

Interoperation with ANDES
-----------------------------------

The interoperation with dynamic simulator invovles both file conversion and data exchange.
In AMS, the built-in interface with ANDES is implemented in :py:mod:`ams.interface`.


File Format Converter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Power flow data is the bridge between scheduling study and dynamics study,
where it defines grid topology and power flow.
An AMS case can be converted to an ANDES case, with the option to supply additional dynamic
data.

.. autofunction:: ams.interface.to_andes
    :noindex:


Data Exchange in Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To achieve scheduling-dynamics cosimulation, it requires bi-directional data exchange between
scheduling and dynamics study.
From the perspective of AMS, two functions, ``send`` and ``receive``, are developed.
The maping relationship for a specific routine is defined in the routine class as ``map1`` and
``map2``.
Additionally, a link table for the ANDES case is used for the controller connections.

Module :py:mod:`ams.interface.Dynamic`, contains the necessary functions and classes for
file conversion and data exchange.

.. autoclass:: ams.interface.Dynamic
    :noindex:
    :members: send, receive

When you use this interface, it automatically picks either the dynamic or static model based on the TDS initialization status.
If the TDS is running, it selects the dynamic model; otherwise, it goes for the static model.
For more details, check out the full API reference or take a look at the source code.

.. note::
      Check ANDES documentation
      `StaticGen <https://docs.andes.app/en/latest/groupdoc/StaticGen.html#staticgen>`_
      for more details about substituting static generators with dynamic generators.