.. _routine:

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
                                  e_str = 'gs @ cp.multiply(ug, pru) - dud')
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
strings in the ``e_str`` argument. ``e_str`` uses **canonical CVXPY syntax**:
call ``cp.multiply(a, b)``, ``cp.sum(x)``, ``cp.power(x, n)`` etc. directly.
Before evaluation, :py:class:`ams.core.symprocessor.SymProcessor` only
resolves bare symbol names (``pg``, ``Cft``, ``rate_a``, …) to the
underlying CVXPY ``Variable`` / ``Parameter`` / sparse-matrix objects of
the host routine. It does **not** rewrite function names: ``cp.*`` calls
must be written explicitly.

.. note::
   Prior to v1.2.2, AMS rewrote ``mul(...)`` → ``cp.multiply(...)``,
   bare ``sum(...)`` → ``cp.sum(...)``, and ``a dot b`` → ``a * b``
   automatically. That rewrite layer has been removed — see the v1.2.2
   migration table in :ref:`ReleaseNotes`. User customizations
   (``addConstrs(e_str=...)`` / ``obj.e_str += '...'``) that still use
   the old vocabulary will raise ``NameError`` at eval time.

Multiplication is the easiest place to introduce silent bugs because there are
**three semantically distinct operations** that all read like "multiply":

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Notation in ``e_str``
     - Meaning
     - When to use
   * - ``a @ b``
     - Matrix / matrix-vector multiply
     - Topology matrix times a stacked vector, e.g. ``Cft @ pl``,
       ``PTDF @ p``.
   * - ``cp.multiply(a, b)``
     - Element-wise multiply
     - Per-device scaling, e.g. ``cp.multiply(ug, pg)`` to gate
       generator output by its commitment. Broadcasts a scalar / 1-D
       parameter against a vector variable.
   * - ``2 * x``, ``coeff * y``
     - Scalar multiply
     - A literal number or a 0-D parameter scaling an expression.
       CVXPY accepts ``*`` here.

**Bare ``a * b`` between two identifiers is not rewritten and will be passed
to CVXPY as-is.** CVXPY will then either raise a shape error or, worse,
emit a deprecation warning and silently perform matrix multiplication. Always
write ``@`` or ``cp.multiply()`` explicitly.

Other canonical-CVXPY constructs in ``e_str``:

* Reductions and atoms — ``cp.sum(x)``, ``cp.norm(x)``,
  ``cp.pos(x)``, ``cp.square(x)``, ``cp.power(x, n)``,
  ``cp.vstack(...)``, ``cp.hstack(...)``, ``cp.maximum(...)``,
  ``cp.minimum(...)``, ``cp.quad_form(...)``,
  ``cp.sum_squares(...)``, ``cp.diag(...)`` — call directly.
* Comparisons ``... == 0`` and ``... <= 0`` define equality and inequality
  constraints respectively (set via the ``is_eq`` flag on ``Constraint``).
* Powers use ``**`` (e.g. ``vmax**2``).

.. warning::
   Routine symbol names (``Var``, ``RParam``, ``Service``,
   ``Expression``, ``Constraint``) may not collide with CVXPY atom
   names (``sum``, ``multiply``, ``vstack``, ``hstack``, ``power``,
   ``norm``, ``pos``, ``neg``, ``square``, ``quad_form``,
   ``sum_squares``, ``diag``, ``maximum``, ``minimum``, ``abs``,
   ``exp``, ``log``, ``sqrt``, ``inv_pos``). The codegen raises an
   explanatory error on collision. Without this guard, a routine that
   declared a symbol named ``sum`` would have its eval-fallback path
   silently rewrite a user-appended ``sum(...)`` into
   ``self.om.sum(...)``.

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

Two execution paths: codegen vs ``eval``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Internally, two execution paths from ``e_str`` to a CVXPY object live
side by side:

- **Codegen path** (fast, default for source-defined items). At
  ``init()`` time, :py:meth:`ams.routines.routine.RoutineBase._link_pycode`
  loads the per-class pycode and wires each item's ``e_fn`` from a
  named callable in that module. ``parse()`` and ``evaluate()`` then
  just invoke the callable with a :py:class:`ams.core.routine_ns.RoutineNS`
  proxy; no regex, no ``eval``.

- **Eval-fallback path** (used for items the codegen doesn't cover).
  At ``parse()`` time, :py:func:`ams.opt._runtime_eval.eval_e_str`
  resolves bare symbol names (``pg`` → ``r.pg``, ``Cft`` → ``r.Cft``,
  …) via a single regex pass and ``eval``\ s the result against a
  ``RoutineNS`` proxy. Function names (``cp.sum``, ``cp.multiply``,
  …) are not rewritten — author canonical CVXPY in ``e_str`` and the
  helper passes them through.

Both paths must produce the same CVXPY object given the same ``e_str``
— the codegen is the AOT-compiled version of the eval-fallback pipeline.

Which path runs is decided per-item by ``_link_pycode``:

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Item state
     - Path
     - Why
   * - In source code, no runtime mutation
     - codegen
     - The pristine ``e_str`` matches what the cache was generated
       from; ``e_fn`` is wired from the cache.
   * - Added at runtime via ``addConstrs`` / ``addExpressions`` /
       ``addExprcs``
     - eval
     - The cache (always generated against a pristine instance —
       see :ref:`pycode-pristine-invariant` below) doesn't know
       about runtime additions.
   * - ``e_str`` reassigned post-construction (e.g.
       ``obj.e_str += '+ ...'``)
     - eval
     - The descriptor mutex on ``e_str`` / ``e_fn`` marks the item
       ``_e_dirty``; ``_link_pycode`` skips wiring so the user's new
       ``e_str`` flows through ``parse()``.

Authors and users can introspect which path is in effect:

- :py:attr:`ams.opt.OptzBase.formulation_source` — per-item, returns
  ``'codegen' | 'eval' | 'manual' | 'pending'``.
- :py:meth:`ams.routines.routine.RoutineBase.formulation_summary` —
  full table.
- An INFO-level log line ``<RoutineName> formulation: codegen=X/Y, …``
  is emitted on every ``init()``.

.. _pycode-pristine-invariant:

The pristine-source invariant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``~/.ams/pycode/<routine>.py`` is **always** a faithful representation
of the routine source code, never of any user customization. To
guarantee this, ``_link_pycode`` runs codegen against a pristine
routine instance fetched from a per-process singleton
``ams.System(no_input=True)`` (see
:py:func:`ams.prep._get_pristine_system`), never against the user's
``self``. A ``pristine = True`` marker in the generated module's
header acts as a cache-validation tripwire — caches written by older
AMS versions (which codegen'd against the live instance) lack the
marker and are auto-invalidated on next read.

This means a user can do::

    sp = ams.load(...)
    sp.DCOPF.obj.e_str += '+ extra_term'
    sp.DCOPF.run(...)                        # uses customization

    sp0 = ams.load(...)                       # fresh instance
    sp0.DCOPF.run(...)                        # uses original formulation

without ``sp``'s mutation leaking into ``sp0``. See
``examples/ex8.ipynb`` for a working sp1 / sp2 / sp3 walkthrough.

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