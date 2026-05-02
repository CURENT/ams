.. _migration_cvxpy_namespace:

Migration: canonical CVXPY in ``e_str`` (v1.2.2)
==================================================

v1.2.2 removes AMS's historical function-name rewrite layer for
``e_str`` strings. ``mul``, ``multiply``, ``sum``, and the ``a dot b``
binary syntax are no longer translated into their ``cp.*`` / ``*``
equivalents — author canonical CVXPY directly. This page is a
field guide for users who maintain external routines or post-init
``e_str`` customizations; built-in routines were migrated in
PR #244 and need no further action.

What changed and why
--------------------

Prior versions ran every ``e_str`` through a regex pre-rewrite that
mapped a small AMS DSL onto the CVXPY namespace:

.. code-block:: text

   mul(a, b)        →  cp.multiply(a, b)
   multiply(a, b)   →  cp.multiply(a, b)
   sum(x)           →  cp.sum(x)
   a dot b          →  a * b

The DSL was inherited from AMS's ANDES-sympy ancestry, when the
regex layer was the only path from a symbolic atom to runnable
code. After the v1.2.2 codegen rewrite (PR #242), generated pycode
imports ``import cvxpy as cp`` directly, so a bare ``cp.multiply``
already resolves correctly through Python's normal name lookup.
The DSL became maintenance burden with no remaining payoff:

- Every new CVXPY atom (e.g. ``cp.huber``) required an entry in
  two parallel allowlists. A user who reached for an unlisted
  atom got a silent ``NameError``, even though CVXPY itself
  supports it.
- ``dot`` was a numpy false-friend. ``np.dot`` is matrix multiply,
  but AMS's ``dot`` rewrote to ``*`` (element-wise). A
  power-systems engineer fluent in numpy reading ``b dot Pg``
  reasonably expected matmul.
- ``mul`` saved three characters versus ``cp.multiply``. That was
  the entirety of its value.

Removing the rewrite layer means every ``e_str`` reads as canonical
CVXPY code that any reader can pick up cold.

Substitution table
------------------

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Old (now broken)
     - New (canonical)
     - Notes
   * - ``mul(a, b)``
     - ``cp.multiply(a, b)``
     - Element-wise multiply.
   * - ``multiply(a, b)``
     - ``cp.multiply(a, b)``
     - Same.
   * - ``sum(x)``
     - ``cp.sum(x)``
     - Reduction over a vector.
   * - ``a dot b``
     - ``a * b``
     - Scalar-times-expression.

The 14 zero-use defensive aliases (``vstack``, ``norm``, ``pos``,
``power``, ``sign``, ``maximum``, ``minimum``, ``square``,
``quad_over_lin``, ``diag``, ``quad_form``, ``sum_squares``,
``var``, ``const``, ``problem``) are also gone. None of them had
any caller in ``ams/routines/``, ``tests/``, or ``examples/``;
write ``cp.<name>(...)`` instead.

A user customization that still uses the old vocabulary raises
``NameError`` at the first call to ``init()`` (codegen path) or
``evaluate()`` (eval-fallback path).

Worked examples (from PR #244)
------------------------------

Pulled directly from ``ams/routines/rted.py``:

**SFR reserve balance** —

.. code-block:: python

   # Before
   self.rbu.e_str = 'gs @ mul(ug, pru) - dud'

   # After
   self.rbu.e_str = 'gs @ cp.multiply(ug, pru) - dud'

**Reserve source bounds** —

.. code-block:: python

   # Before
   self.rru.e_str = 'mul(ug, (pg + pru)) - mul(ug, pmaxe)'

   # After
   self.rru.e_str = 'cp.multiply(ug, (pg + pru)) - cp.multiply(ug, pmaxe)'

**Quadratic cost with time scaling** — illustrates ``dot`` →
``*`` and ``sum`` → ``cp.sum`` together:

.. code-block:: python

   # Before
   cost = 't**2 dot sum(mul(c2, pg**2)) + sum(mul(ug, c0))'
   cost += f'+ t dot sum({_to_sum})'

   # After
   cost = 't**2 * cp.sum(cp.multiply(c2, pg**2)) + cp.sum(cp.multiply(ug, c0))'
   cost += f'+ t * cp.sum({_to_sum})'

**ESD1 SOC balance** — from ``ESD1PBase`` in ``ams/routines/rted.py``:

.. code-block:: python

   # Before
   SOCb = 'mul(En, (SOC - SOCinit)) - t dot mul(EtaC, pce)'
   SOCb += '+ t dot mul(REtaD, pde)'

   # After
   SOCb = 'cp.multiply(En, (SOC - SOCinit)) - t * cp.multiply(EtaC, pce)'
   SOCb += '+ t * cp.multiply(REtaD, pde)'

Customization patterns
----------------------

Both supported customization entry points expect canonical CVXPY:

.. code-block:: python

   # Append a quadratic penalty to an existing objective
   sp.RTED.obj.e_str += '+ 0.01 * cp.sum(cp.square(pg - pg0))'

   # Add a new constraint at runtime
   sp.RTED.addConstrs(
       name='pg_band',
       e_str='cp.abs(pg - pg0) - 0.1 * pmax',
   )

The ``cp.`` prefix is required even after the customization;
there is no implicit ``from cvxpy import *`` in the eval scope.

Reserved-name guard
-------------------

A new check rejects routine symbol names that collide with a CVXPY
atom (e.g. defining ``self.sum = Var(...)``). Without the guard, a
post-init customization like ``addConstrs(e_str='cp.sum(sum)')``
would silently rewrite the inner ``sum`` to ``r.sum`` (the user's
``Var``) and resolve correctly, but a bare ``sum(...)`` in the
same routine's source would resolve to the user's ``Var`` instead
of ``cp.sum``. The guard raises ``ValueError`` at codegen time
naming the offending symbol.

Reserved names are the lowercase CVXPY atoms: ``sum``, ``multiply``,
``vstack``, ``hstack``, ``power``, ``norm``, ``pos``, ``neg``,
``square``, ``quad_form``, ``sum_squares``, ``diag``, ``maximum``,
``minimum``, ``abs``, ``exp``, ``log``, ``sqrt``, ``inv_pos``.
None of the built-in routines collide; an ``ams/routines/`` AST
sweep landed with PR #244 confirms this.

``formulation_source`` and the eval-fallback rename
---------------------------------------------------

The :py:attr:`ams.opt.OptzBase.formulation_source` value
``'sub_map'`` is renamed to ``'eval'``. The old name referred to
the deleted ``SymProcessor.sub_map`` regex pipeline; the new name
reflects the live implementation —
:py:func:`ams.opt._runtime_eval.eval_e_str` and its numeric
twin ``eval_e_str_numeric``.

Update any introspection code:

.. code-block:: python

   # Before
   if item.formulation_source == 'sub_map':
       ...

   # After
   if item.formulation_source == 'eval':
       ...

The companion INFO log line on every ``init()`` is renamed in
lockstep:

.. code-block:: text

   # Before
   <DCOPF> formulation: codegen=14/17, sub_map(customized)=1, sub_map(added)=2

   # After
   <DCOPF> formulation: codegen=14/17, eval(customized)=1, eval(added)=2

The :py:meth:`ams.routines.routine.RoutineBase.formulation_summary`
print bucket header changes from ``sub_map`` to ``eval`` for the
same reason.

``SymProcessor.sub_map`` and ``val_map`` removal
------------------------------------------------

Both attributes are deleted. They populated dicts that fed the
legacy regex-rewrite + ``eval`` pipeline; nothing reads them now
that the eval-fallback helper performs symbol resolution
inline. Anyone introspecting the old structures should switch to:

- :py:attr:`ams.opt.OptzBase.formulation_source` for per-item
  provenance.
- The routine's own ``vars`` / ``rparams`` / ``services`` /
  ``exprs`` / ``constrs`` registries for symbol enumeration.
- :py:class:`ams.core.routine_ns.RoutineNS` (and
  ``NumericRoutineNS``) for the live attribute-resolution proxies
  the helper uses internally.

The ``inputs_dict``, ``services_dict``, ``tex_names``, and
``tex_map`` attributes on ``SymProcessor`` are unchanged — they
still feed LaTeX rendering and downstream consumers.

``Constraint.is_eq`` retired — embedded-operator LHS-zero ``e_str``
-------------------------------------------------------------------

The ``is_eq`` parameter on :py:class:`ams.opt.Constraint` and
:py:meth:`ams.routines.routine.RoutineBase.addConstrs` is removed.
Every ``e_str`` must embed the relational operator and follow the
LHS-zero authoring discipline — all terms on the left, ``<= 0``,
``== 0``, or ``>= 0`` on the right:

.. code-block:: python

   # Before — operator implicit, polarity in is_eq flag
   self.pglb = Constraint(name='pglb', e_str='-pg + pmine')
   self.pb = Constraint(name='pb', e_str=pb, is_eq=True)
   sp.RTED.addConstrs(name='cap', e_str='pg - pmax', is_eq=False)

   # After — operator embedded, single source of truth in e_str
   self.pglb = Constraint(name='pglb', e_str='-pg + pmine <= 0')
   self.pb = Constraint(name='pb', e_str=pb + ' == 0')
   sp.RTED.addConstrs(name='cap', e_str='pg - pmax <= 0')

**Why LHS-zero rather than ``pg <= pmax``-style?** ``Constraint.v``
returns ``optz._expr.value`` — the CVXPY-canonical LHS expression.
With LHS-zero authoring, ``.v`` is the slack from zero (negative =
respected, positive = violated, magnitude = by how much). Both
authoring forms preserve the invariant since CVXPY normalizes
``a <= b`` and ``b >= a`` to ``_expr = a - b`` internally — but
the LHS-zero form makes the diagnostic intent explicit in the
source text.

Strict ``<`` and ``>`` are not accepted by CVXPY anywhere
(``NotImplementedError``); only ``<=``, ``==``, ``>=`` may end an
``e_str``.

A ``Constraint`` whose ``e_str`` (or ``e_fn`` body) does not
produce a ``cp.constraints.Constraint`` raises ``TypeError`` at
evaluate time with a message naming the embedded-operator forms
expected.

**Custom callables.** Authors who supply their own ``e_fn`` (instead
of ``e_str``) must now return a fully-formed
``cp.constraints.Constraint`` — the codegen convention of returning
a bare LHS expression and letting ``Constraint.evaluate`` apply
``<= 0`` / ``== 0`` is gone.

**Cached pycode is auto-invalidated.** ``PYCODE_FORMAT_VERSION`` is
bumped in lockstep with this change, so any
``~/.ams/pycode/<routine>.py`` written by a pre-retirement AMS is
rejected on first read and regenerated. No manual
``ams prep --force`` needed after upgrade.

See also
--------

- :ref:`routine` — the "Expression Notation in ``e_str``" section
  covers canonical CVXPY syntax in depth.
- ``examples/ex8.ipynb`` — sp1 / sp2 / sp3 walkthrough of post-init
  customization patterns under the canonical rules.
- :ref:`ReleaseNotes` — v1.2.2 entry, "Migration — ``e_str``
  authoring contract".
