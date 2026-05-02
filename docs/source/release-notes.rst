.. _ReleaseNotes:

=============
Release notes
=============

The APIs before v3.0.0 are in beta and may change without prior notice.

v1.2
==========

v1.2.3 (unreleased)
----------------------

**Tightening the eval-fallback path:**

- :class:`ams.opt.Constraint` now validates at parse time that
  ``e_str`` ends with the LHS-zero shape (``<= 0`` / ``== 0`` /
  ``>= 0``). Authoring ``pg <= pmax`` instead of
  ``pg - pmax <= 0`` previously solved correctly but produced a
  silently-wrong :pyattr:`OptzBase.e` (numpy bool array instead of
  the LHS slack); the new :func:`ams.opt._runtime_eval.assert_constraint_lhs_zero`
  guard surfaces the mismatch immediately with an actionable error.
- :func:`ams.opt._runtime_eval.eval_e_str` now logs a per-item
  ``WARNING`` when an eval-fallback constraint/objective/expression
  yields a non-DCP CVXPY object (e.g. ``cp.multiply(pg, pg)`` instead
  of ``cp.square(pg)``). The codegen path is unaffected; this
  closes the visibility gap on user customizations and ``addConstrs``
  added by the cvxpy-namespace passthrough.
- :data:`ams.prep.generator.RESERVED_CVXPY_ATOM_NAMES` is now
  derived at import time from ``cvxpy.atoms`` (still a strict
  superset of the static fallback). New atoms in future CVXPY
  releases are guarded automatically.

v1.2.2 (unreleased)
----------------------

**Migration — ``Constraint.is_eq`` retired:**

The ``is_eq`` parameter on :class:`ams.opt.Constraint` and
:meth:`ams.routines.routine.RoutineBase.addConstrs` is removed.
Every ``Constraint`` ``e_str`` must now embed the relational
operator in the LHS-zero form: ``'<LHS> <= 0'``, ``'<LHS> == 0'``,
or ``'<LHS> >= 0'``. The CVXPY canonicalization of every
inequality to ``lhs - rhs <= 0`` (regardless of ``<=`` vs ``>=``
direction) means ``Constraint.v`` keeps reporting slack-from-zero
uniformly.

Update any user routines or runtime customizations:

.. code-block:: python

   # Before
   self.pglb = Constraint(name='pglb', e_str='-pg + pmine')
   self.pb = Constraint(name='pb', e_str=pb, is_eq=True)
   sp.RTED.addConstrs(name='cap', e_str='pg - pmax', is_eq=False)

   # After
   self.pglb = Constraint(name='pglb', e_str='-pg + pmine <= 0')
   self.pb = Constraint(name='pb', e_str=pb + ' == 0')
   sp.RTED.addConstrs(name='cap', e_str='pg - pmax <= 0')

A ``Constraint`` whose ``e_str`` (or ``e_fn`` body) does not
produce a ``cp.constraints.Constraint`` now raises ``TypeError``
at evaluate time with a message naming the embedded-operator
forms expected. The codegen and eval-fallback paths share this
behavior.

See :ref:`migration_cvxpy_namespace` for the full rationale and
worked examples.

**Migration — ``e_str`` authoring contract:**

Routine ``e_str`` strings (built-in routines and any user customization
via ``addConstrs`` / ``obj.e_str += '...'``) must now use the canonical
CVXPY namespace explicitly. The historical function-name rewrite layer
that translated ``mul(...)``, ``multiply(...)``, ``sum(...)``, and
``a dot b`` into their ``cp.*`` / ``*`` equivalents has been removed.

Update any user customizations:

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
     - Scalar-times-expression. Was already a thin alias.

Routine symbol names (``Var``, ``RParam``, ``Service``, ``Expression``,
``Constraint``) may no longer match a CVXPY atom name (e.g. ``sum``,
``multiply``, ``vstack``, ``power``). The codegen path raises an
explanatory error if a collision is detected. This prevents a routine
from silently shadowing ``cp.X`` in the eval-fallback rewrite.

A small behavior change rides with the eval-fallback unification:
``Constraint.e_str`` may now contain a relational operator
(``pg <= pmax`` etc.). Previously ``Constraint.parse()`` appended
``' == 0'`` / ``' <= 0'`` unconditionally and an embedded operator
would have been a syntax-level "double op" at evaluate time. The
runtime now dispatches on whether the eval result is already a
``cp.constraints.Constraint`` and skips the wrap when it is.

The ``SymProcessor.sub_map`` and ``SymProcessor.val_map``
attributes are removed. Both fed the legacy regex-rewrite +
``eval`` pipeline that the cvxpy-namespace passthrough has fully
replaced — symbol resolution now goes through
``ams.opt._runtime_eval`` plus :class:`RoutineNS` /
:class:`NumericRoutineNS`. Anyone introspecting those dicts
should switch to ``formulation_source`` plus the routine's
``vars`` / ``rparams`` / ``services`` / ``exprs`` / ``constrs``
registries.

The :py:attr:`ams.opt.OptzBase.formulation_source` value
``'sub_map'`` is renamed to ``'eval'``. The string referred to
the (now-deleted) ``SymProcessor.sub_map`` regex pipeline; the
new name reflects the live implementation —
``ams.opt._runtime_eval.eval_e_str`` /
``eval_e_str_numeric``. Code that compared
``item.formulation_source == 'sub_map'`` must update to
``'eval'``. The corresponding INFO log keys (formerly
``sub_map(customized)`` / ``sub_map(added)``) and
``formulation_summary`` print bucket are renamed in lockstep.

**New features:**

- **Opt-layer codegen.** Routine constraints, expressions, and
  objectives written as ``e_str`` strings are now compiled once to
  named Python callables in ``~/.ams/pycode/<routine>.py``. The
  runtime path imports those callables instead of regex-rewriting
  ``e_str`` and ``eval``-ing the result on every routine init.

  Author-facing API is unchanged — keep writing ``e_str``. The
  generated cache refreshes on first init when the routine source
  ``md5`` changes; it can also be regenerated explicitly with the
  new ``ams prep`` CLI:

  .. code-block:: console

     $ ams prep                    # generate for every routine
     $ ams prep --routine DCOPF    # restrict to one
     $ ams prep --force            # regenerate even when md5 matches
     $ ams prep --clean            # wipe ~/.ams/pycode/
     $ ams prep --where            # print the cache path

  See :ref:`routine` for the rendered pipeline. Generated pycode is
  per-user and never shipped in the wheel — analogous to ANDES's
  ``andes prep`` / ``~/.andes/pycode/`` pattern.

- ``Objective.add_term(fn)`` lets a mixin subclass extend a parent
  objective with an extra cost term via callable rather than the
  legacy ``self.obj.e_str += '...'`` string append. Used internally
  by ``ESD1Base`` (charge/discharge cost) and ``RTEDVIS`` (virtual
  inertia/damping cost); available to user-defined mixins too.

- DPP-compliance diagnostic logged at ``OModel.finalize`` time. When
  a routine has parameters but the resulting ``cp.Problem`` is not
  DPP, an info-level log line warns that warm re-solves will
  re-canonicalize. Helps catch accidental non-DPP terms early.

- **Formulation-source introspection.** Three ways to see which
  execution path an opt element runs through after ``init()``:

  - per-item :py:attr:`ams.opt.OptzBase.formulation_source` →
    ``'codegen' | 'eval' | 'manual' | 'pending'``;
  - :py:meth:`ams.routines.routine.RoutineBase.formulation_summary`
    prints a per-item table;
  - an INFO log line on every ``init()``::

       <DCOPF> formulation: codegen=14/17, eval(customized)=1, eval(added)=2

  Useful for confirming a runtime customization
  (``addConstrs``, ``obj.e_str += '...'``) actually reaches the
  optimizer rather than being silently overwritten by the cached
  callable.

**Behavior changes:**

- **Disk cache is always pristine.** The generated pycode at
  ``~/.ams/pycode/<routine>.py`` is now guaranteed to reflect only
  the routine source code, never per-instance customizations. Codegen
  always runs against a fresh ``ams.System`` instance fetched from a
  per-process singleton, never against the user's ``self``. A
  ``pristine = True`` marker in the generated module's header
  auto-invalidates caches written by older AMS versions on next read.
  This means runtime mutations on one ``System`` (``sp.DCOPF.obj.e_str
  += '...'``, ``sp.DCOPF.addConstrs(...)``) cannot leak into another
  freshly-loaded ``System``. See ``examples/ex8.ipynb`` for a working
  sp1 / sp2 / sp3 demonstration.

- ``e_str`` is preserved on items after auto-prep wires their
  ``e_fn`` from pycode. Lets users do
  ``sp.DCOPF.obj.e_str += '+ ...'`` post-init — a documented
  customization pattern that briefly broke when the ``_EFormDescriptor``
  mutex cleared ``e_str`` on internal codegen wiring.

- ``.e`` (the post-solve / debug numerical value of a constraint
  LHS, expression, or objective) now works correctly even when the
  solver bails or hasn't run yet. Resolves through a numeric proxy
  that returns ``Var.v`` (zeros fallback) for unset variables — the
  same semantics the legacy ``val_map``-eval path had.

**Bug fixes:**

- ``bench.harness.measure``: when warmup raises, the failure path now
  preserves the structured tags (``routine``, ``case``, ``solver``,
  ``phase``, ``metadata``) on the returned ``Measurement`` instead of
  dropping them.
- ``LazyImport.__maybe_import_complementary_imports__`` narrows the
  swallowed exception from bare ``except Exception`` to
  ``except ImportError`` and logs at DEBUG, so unexpected failures stop
  being silently hidden.

**Internal:**

- Drop Codecov from CI in favor of Codacy as the single coverage
  provider. Removes ``.codecov.yml``, the ``codecov/codecov-action``
  + ``codecov/test-results-action`` upload steps, the duplicate
  pytest run that produced ``junit.xml``, and the Codecov badge in
  ``README.md``. Renames ``.github/workflows/codecov.yml`` to
  ``coverage.yml`` to match its now-Codacy-only role. The
  ``CODECOV_TOKEN`` repository secret can be deleted.
- Silence Codacy's Bandit findings on the env-capture subprocess calls
  in ``ams/bench/env.py`` via per-call ``# nosec B404``/``# nosec B603``
  markers. Codacy's Bandit engine doesn't honor the
  ``[tool.bandit] skips`` block in ``pyproject.toml`` (it only passes a
  config file when the project's pattern set is empty), so the inline
  markers are what actually apply on Codacy. Local ``bandit -c
  pyproject.toml`` runs continue to honor the skips list.
- ``ams.io.dump``: initialize ``ret`` defensively before the
  format-dispatch ``if/elif`` to silence pylint's
  ``possibly-used-before-assignment``. The earlier ``output_formats``
  membership guard already ensures one branch fires at runtime, so
  ``ret`` was never actually unbound — this is a static-analysis
  cleanup, not a behavior change.
- Add ``encoding='utf-8'`` to text-mode ``open()`` /
  ``Path.read_text`` calls in ``ams.bench.env``, ``ams.bench.__main__``,
  and ``ams.system`` (config-file write). Avoids the platform-default
  encoding foot-gun on Windows.
- ``link_ext_param(model=None)`` and ``Model.set(..., base=None)``:
  document and explicitly drop the API-compat-only arguments so they
  no longer surface as ``unused-argument`` lint findings.
- Bulk-disable noisy pylint patterns in ``.prospector.yaml``
  (``too-many-arguments`` / ``-locals`` / ``-branches``,
  ``use-list-literal`` / ``use-dict-literal``, ``no-else-return``,
  ``import-outside-toplevel``, ``eval-used``) with rationale; replaces
  per-PR Codacy churn on long-standing code.

v1.2.1 (2026-04-29)
----------------------

**Bug fixes:**

- ``ams st`` (a.k.a. ``ams.main.selftest``) now degrades gracefully on
  wheel installs. v1.2.0 correctly excluded the ``tests/`` directory
  from the built wheel, but ``ams st`` was still calling
  ``unittest.TestLoader().discover('<site-packages>/tests')`` and
  crashing with ``ImportError: Start directory is not importable``. The
  command now detects the missing directory, logs a one-line message
  pointing the user to ``pytest`` from a clone, and exits cleanly. From
  a source clone the behavior is unchanged. While in the function,
  also harden the stdout-suppression block: save the caller's stdout
  (not ``sys.__stdout__``) so wrappers like
  ``contextlib.redirect_stdout`` are preserved, restore + close the
  ``/dev/null`` handle in a ``finally``, and gate the redirect on
  ``logger.handlers`` being non-empty (replacing a try/except IndexError)

**Improvements:**

- Add top-level ``ams --version`` flag that prints the AMS version
  (``ams X.Y.Z``), following the standard CLI convention used by
  ``pip``, ``pytest``, ``git``, and others. ``ams misc --version`` is
  unchanged and still prints the multi-line dependency block (Python,
  andes, numpy, cvxpy, solvers) for bug reports
- ``RParam.v``: pass scipy.sparse values through untouched on the
  ``is_ext`` branch (caller-supplied ``v=...``). PR #233 fixed
  ``MParam.v`` but missed this parallel path; now the no-auto-densify
  contract is consistent across ``RParam`` and ``MParam``
- ``NumOpDual.v0``: pass scipy.sparse outputs through untouched when
  ``array_out=True``, mirroring the ``NumOp.v0`` fix from PR #233.
  Previously sparse results were wrapped in a 1-element object-dtype
  ``ndarray``, which broke downstream ``csr_matrix`` conversion. Audit
  of all other ``v0`` overrides (``NumHstack``, ``ZonalSum``,
  ``VarSelect``, ``VarReduction``, ``RampSub``) confirmed they all
  construct results via numpy ops that already return ``ndarray``, so
  no parallel fix was needed there
- ``MParam.v`` no longer auto-densifies sparse-stored matrices on every
  property access. The underlying scipy.sparse object is now returned
  as-is; a new ``MParam.dense()`` method materializes a dense
  :class:`numpy.ndarray` when one is genuinely required. ``.shape`` and
  ``.n`` consult the underlying value directly without densifying.
  Together with the ``sparse=True`` declarations on the matrix
  ``RParam`` consumers in ``DCPF`` / ``DCOPF`` / ``DCOPF2``, this lets
  routines actually pass sparse matrices to CVXPY instead of silently
  densifying them on every parameter set
- ``MatProcessor.build_ptdf`` / ``build_lodf`` now finalize the result
  as ``csr_matrix`` rather than leaving it as ``lil_matrix``. ``lil`` is
  the right format for incremental row assignment during construction
  but is poorly supported by downstream consumers (CVXPY, scipy ``COO``
  conversion); freezing to ``csr`` at the end is the canonical pattern
- ``NumOp.v0``: pass scipy.sparse outputs through untouched when
  ``array_out=True``. Previously sparse results were wrapped in a
  1-element :class:`numpy.ndarray` of object dtype, which broke the
  subsequent ``csr_matrix`` conversion in :meth:`NumOp.v` for sparse
  ``NumOp``\ s such as ``DCOPF2.PTDFt``
- ``MatProcessor.build_lodf``: replace the dense ``np.ones`` + ``np.tile``
  column-broadcast inside the chunk loop with sparse column scaling
  that divides each column by ``(1 - h_chunk[j])``, zeroing columns
  where the denominator is zero to preserve the prior ``safe_div``
  behavior. Removes a transient ``nbranch × step`` dense allocation
  that reached ~1 GB on ~70k-bus grids. Numerical output is unchanged.
- ``MatProcessor.build_cg``: replace the O(ng²) ``list.index`` lookup
  with ``StaticGen.idx2uid`` for O(ng) total cost. Matters as
  generator counts grow; modest at current case sizes.
- ``MatProcessor.build_ptdf``: factorize the reduced bus susceptance
  matrix once via :func:`scipy.sparse.linalg.splu` and reuse the
  factorization across all line chunks. Removes the dense full-``Bbus``
  solver path, which materialized an ``nb × nb`` dense matrix and was
  unusable on grids beyond a few thousand buses. Large-case PTDF builds
  are noticeably faster (e.g. ~2.5× on a 2000-bus case in incremental
  mode); numerical output is unchanged. The ``use_umfpack`` keyword is
  retained for backward compatibility but is now a no-op.

v1.2.0 (2026-04-23)
----------------------

**Breaking changes:**

- Requires ANDES >= 2.0.0
- Minimum Python version raised to 3.11 (supported: 3.11, 3.12, 3.13)
- Installation now uses pyproject extras instead of ``requirements*.txt``
  files. Development install: ``pip install -e '.[dev,nlp]'``; docs:
  ``pip install -e '.[doc]'`` (single quotes are required in ``zsh``
  because ``[...]`` is a glob pattern)
- In routine ``e_str`` expressions, multiplication must be explicit:
  use ``@`` for matrix-vector, ``mul()`` for element-wise, or ``*`` for
  scalar. The previous automatic ``word*word → @`` rewrite has been
  removed. See :doc:`/modeling/routine` for details

**New:**

- Performance benchmark suite under ``ams.bench``. Run
  ``python -m ams.bench`` to reproduce timings across standard cases
  and routines; baselines in ``bench/baselines/``

**Improvements:**

- ``import ams`` is faster and no longer instantiates an ANDES
  ``System`` at import time
- CLI: ``ams`` without a subcommand now prints help cleanly; unknown
  commands produce a clear error instead of a traceback
- Loading a case produces far fewer warnings (retired config keys are
  now silently purged instead of flagged on every startup)

**Internal:**

- Build system migrated from ``versioneer`` to ``setuptools-scm``;
  this is the first release produced by the new pipeline

v1.0
==========

v1.1.1 (2025-12-23)
----------------------

- Include objective function value in exported report
- Add PTDF-based routines: ``RTED2``, ``RTED2DG``, ``RTED2ES``
- Add routine ``RTEDESP`` for price run of ``RTED2`` and ``RTED2ES``
- Remove attribute `info` from class RoutineBase and all derived routines
- Fix charging/discharging duration constraints in ``RTEDES``
- Refactor routine architecture to use cooperative multiple inheritance (MRO)
  for better component mixins
- Refactor `DCPF1.run` and `OPF.run`

v1.1.0 (2025-12-18)
----------------------

- Add parameter check to correct ESD1 associated GCost
- Add ESD1 params: `SOCend`, `cesdc`, `cesdd`, `tdc`, `tdd`, `tdc0`, `tdd0`
- Revise ESD1 related routines: `RTEDES`, `EDES`, and `UCES`

v1.0.16 (2025-09-28)
----------------------

- Add GitHub action to automatically upload release to Zenodo

v1.0.15 (2025-09-28)
----------------------

- Fix ``DCOPF2.pb`` to use PTDF-based formulation
- Add a demo to compare ``DCOPF2`` and ``DCOPF``, see
  ``examples/demonstration/demo_dcopf.ipynb``
- In DC type routines, set `vBus` value as 1 for placeholder
- Include Summary and Objective value in exported JSON

v1.0.14 (2025-08-29)
----------------------

- Add supported Python versions in README


v1.0.13 (2025-08-18)
----------------------

- Add methods ``export_npz``, ``load_npz``, ``load_csv`` in ``MatProcessor``
  to save and load matrices from files
- Add methods ``export_json``, ``load_json`` in ``RoutineBase``
  to save and load routine results from JSON files

v1.0.12 (2025-05-29)
----------------------

- Add RParam pd and qd in ``DCPF1`` for easy access to load
- Bug fix in ``RoutineBase.export_csv`` when path is specified
- Fix bug in ``io.matpower.system2mpc`` with multiple PQ at one bus

v1.0.11 (2025-05-23)
----------------------

- Refactor Documenter
- Fix bug in documentation building

v1.0.10 (2025-05-23)
----------------------

- Add bus type correction in ``system.System.setup``
- Revise ``io.psse.read`` to complete model Zone when necessary
- Use numerical Area and Zone idx in MATPOWER and PSSE RAW file conversion
- Support JSON format addfile when converting to ANDES case
- Add PSS/E v33 RAW file writer
- In class ``System``, add wrapper methods ``to_mpc``, ``to_m``, ``to_xlsx``,
  ``to_json``, and ``to_raw`` for easier case file export
- Add wrapper routines for PYPOWER: ``DCPF1``, ``PFlow1``, ``DCOPF1``, and ``ACOPF1``
- In routine ``DCOPF`` and its derivatives, add ExpressionCalc ``mu1`` ``mu2`` to
  calculate Lagrange multipliers of line flow limits constraints
- Add wrapper routine ``OPF`` for gurobi_optimods
- Add a demo to show newly added wrapper routines, see
  ``examples/demonstration/demo_wrapper_routines.ipynb``
- Revise ``andes.common.config.Config.update`` to ensure configuration parameters
  are consistently updated in both the object and its internal ``_dict``
- Remove legacy revised PYPOWER module
- Remove function ``shared.ppc2df``

v1.0.9 (2025-04-23)
--------------------

Improve MATPOWER file converter:

- Add a M file case writer
- Include Area and Zone in both MATPOWER read and write

v1.0.8 (2025-04-20)
--------------------

- Run workflow "Publish" only on push tag event
- Include Hawaii synthetic case from
  `Hawaii Synthetic Grid <https://electricgrids.engr.tamu.edu/hawaii40/>`_
- Remove matrices calculation functions in model ``Line``
- Include ``gentype`` and ``genfuel`` when parsing MATPOWER cases
- Fix logging level in ``ACOPF.run``

v1.0.7 (2025-04-14)
--------------------

- Address several wording issues in the documentation
- Switch to ``Area`` from ``Zone`` for zonal calculation
- Extend common parameters in groups ``StaticGen`` and ``StaticLoad`` with ``area``
- Set case ``pjm5bus_demo.xlsx`` as a all-inclusive case
- Include module ``MatProcessor`` in the API documentation
- Improve Line parameters correction in ``System.setup``
- Make func ``interface._to_andes_pflow`` public
- Discard ``sync_adsys`` step in func ``to_andes_pflow`` to fix mistake in
  parameters conversion
- Update case files

v1.0.6 (2025-04-10)
--------------------

- Enhance handling of Type 1 gencost: Automatically fallback to Type 2 gencost
- Add parameter correction for zero line angle difference

v1.0.5 (2025-04-09)
--------------------

- Include sensitivity matrices calculation demo in documentation
- Add ``DCOPF2``, a PTDF-based DCOPF routine
- Fix bug when update routine parameters before it is initialized

v1.0.4 (2025-04-05)
--------------------

- Fix format in release notes
- Add badges of GitHub relesase and commits in README
- Add a demo to show sensitivity matrices calculation

v1.0.3 (2025-03-17)
--------------------

- Bug fix in function ``interface.parse_addfile``, released in v1.0.3a1

v1.0.2 (2025-02-01)
--------------------

- Enhance the GitHub Actions workflow file
- Deprecate andes logger configuration in ``config_logger``
- Deprecate solver specification in ``demo_ESD1``

v1.0.1 (2025-01-26)
--------------------

Hotfix: removed dependencies on `SCIP` and `pyscipopt` to resolve installation issues

v1.0.0 (2025-01-24)
--------------------

- **Breaking Change**: rename model ``Region`` to ``Zone`` for clarity. Prior case
  files without modification can run into error.
- Fix bugs in ``RTED.dc2ac``
- Minor refacotr ``OptzBase.get_idx`` to reduce duplication
- Rename module ``OptBase`` to ``OptzBase`` for clarity
- Update benchamrk figure in README
- Set ANDES requirement to v1.9.3
- Deprecate method ``get_idx`` and suggest using ``get_all_idxes`` instead
- Remove module ``benchmarks.py`` and its tests for simplicity

Pre-v1.0
==========

v0.9.13 (2024-12-05)
--------------------

- Add a step to report in ``RoutineBase.run``
- Add more tests to cover DG and ES related routines
- Improve formulation for DG and ESD involved routines
- Improve module ``Report`` and method ``RoutineBase.export_csv``
- Support ``TimedEvent`` in ANDES case conversion
- Add Var ``vBus`` in ``DCOPF`` for placeholder

v0.9.12 (2024-11-23)
--------------------

- Refactor ``OModel.initialized`` as a property method
- Add a demo to show using ``Constraint.e`` for debugging
- Fix ``opt.omodel.Param.evaluate`` when its value is a number
- Improve ``opt.omodel.ExpressionCalc`` for better performance
- Refactor module ``opt``
- Add class ``opt.Expression``
- Switch from PYPOWER to ANDES in routine ``PFlow``
- Switch from PYPOWER to regular formulation in routine ``DCPF``
- Refactor routines ``DCPF`` and ``DCOPF``
- In ``RDocumenter``, set Srouce to be owner if there is no src
- Specify ``multiprocess<=0.70.16`` in requirements as 0.70.17 does not support Linux

RC1
~~~~
- Reset setup.py to ensure compatibility

v0.9.11 (2024-11-14)
--------------------

- Add pyproject.toml for PEP 517 and PEP 518 compliance
- Add model ``Jumper``
- Fix deprecation warning related to ``pandas.fillna`` and ``newshape`` in NumPy
- Minor refactor on solvers information in the module ``shared``
- Change default values of minimum ON/OFF duration time of generators to be 1 and 0.5 hours
- Add parameter ``uf`` for enforced generator on/off status
- In servicee ``LoadScale``, consider load online status
- Consider line online status in routine ``ED``
- Add methods ``evaluate`` and ``finalize`` in the class ``OModel`` to handle optimization 
  elements generation and assembling
- Refactor ``OModel.init`` and ``Routine.init``
- Add ANDES paper as the citation file for now
- Add more routine tests for generator trip, line trip, and load trip
- Add a README to overview built-in cases
- Rename methods ``v2`` as ``e`` for classes ``Constraint`` and ``Objective``
- Add benchmark functions
- Improve the usage of ``eval`` in module ``omodel``
- Refactor module ``interop.andes`` as module ``interface`` for simplicity

v0.9.10 (2024-09-03)
--------------------

Hotfix of import issue in ``v0.9.9``.

- In module ``MatProcessor``, add two parameters ``permc_spec`` and ``use_umfpack`` in function ``build_ptdf``
- Follow RTD's deprecation of Sphinx context injection at build time
- In MATPOWER conversion, set devices name as None
- Skip macOS tests in azure-pipelines due to failure in fixing its configuration
- Prepare to support NumPy v2.0.0, but solvers have unexpected behavior
- Improve the logic of setting ``Optz`` value
- Support NumPy v2.0.0

v0.9.9 (2024-09-02)
-------------------

**NOTICE: This version has known issues and has been yanked on PyPI.**

v0.9.8 (2024-06-18)
-------------------

- Assign ``MParam.owner`` when declaring
- In ``MatProcessor``, improve ``build_ptdf`` and ``build_lodf`` to allow partial building and
  incremental building
- Add file ``cases/matpower/Benchmark.json`` for benchmark with MATPOWER
- Improve known good results test
- Minor fix in ``main.py`` selftest part
- Set dependency NumPy version to be <2.0.0 to avoid CVXPY compatibility issues

v0.9.7 (2024-05-24)
-------------------

This patch release add the Roadmap section in the release notes, to list out some potential features.
It also drafts the EV Aggregation model based on the state space modelg, but the finish date remains unknown.

References:

[1] J. Wang et al., "Electric Vehicles Charging Time Constrained Deliverable Provision of Secondary
Frequency Regulation," in IEEE Transactions on Smart Grid, doi: 10.1109/TSG.2024.3356948.

- Fix OTDF calculation
- Add parameter ``dtype='float64'`` and ``no_store=False`` in ``MatProcessor`` PTDF, LODF, and OTDF
  calculation, to save memory
- Add placeholder parameter ``Bus.type``

v0.9.6 (2024-04-21)
-------------------

This patch release refactor and improve ``MatProcessor``, where it support PTDF, LODF,
and OTDF for static analysis.

The reference can be found online "PowerWorld > Web Help > Sensitivities > Line
Outage Distribution Factors".

- Refactor DCPF, PFlow, and ACOPF
- Add a loss factor in ``RTED.dc2ac``
- Add ``DCOPF.dc2ac``
- Fix OModel parse status to ensure no_parsed params can be updated
- Fix and rerun ``ex2``
- Format ``Routine.get`` return type to be consistent with input idx type
- Remove unused ``Routine.prepare``
- Refactor ``MatProcessor`` to separate matrix building
- Add Var ``plf`` in ``DCPF``, ``PFlow``, and ``ACOPF`` to store the line flow
- Add ``build_ptdf``, ``build_lodf``, and ``build_otdf``
- Fix ``Routine.get`` to support pd.Series type idx input
- Reserve ``exec_time`` after ``dc2ac``
- Adjust kloss to fix ``ex2``

v0.9.5 (2024-03-25)
-------------------

- Add more plots in ``demo_AGC``
- Improve line rating adjustment
- Adjust static import sequence in ``models.__init__.py``
- Adjust pjm5bus case line rate_a
- Fix formulation of constraint line angle diff
- Align slack bus angle to zero in ``DCOPF``
- Align StaticGen idx sequence with converted MATPOWER case
- Fix several issues in MATPOWER converter

v0.9.4 (2024-03-16)
-------------------

- Add Var ``pi`` and ExpressionCalc ``pic`` to store the dual of constraint power balance
- Add Param ``M`` and ``D`` to model ``REGCV1``
- Add CPS1 score calculation in ``demo_AGC``

v0.9.3 (2024-03-06)
-------------------

- Major improvemets on ``demo_AGC``
- Bug fix in ``RTED.dc2ac``

v0.9.2 (2024-03-04)
-------------------

- Add ``demo_AGC`` to demonstrate detailed secondary frequency regulation study
- Add ``ExpressionCalc`` to handle post-solving calculation
- Rename ``type='eq'`` to ``is_eq=False`` in ``Constraint`` to avoid overriding built-in attribute
- Several formatting improvements

v0.9.1 (2024-03-02)
-------------------

- Change sphinx extension myst_nb to nbsphinx for math rendering in ``ex8``
- Improve ``symprocessor`` to include routine config
- Add config to Routine reference
- Fix symbol processor issue with power operator

v0.9.0 (2024-02-27)
-------------------

- Add ``ex8`` for formulation customization via API
- Improve Development documentation
- Fix ``addService``, ``addVars``
- Rename ``RoutineModel`` to ``RoutineBase`` for better naming
- Fix ANDES file converter issue
- Initial release on conda-forge

v0.8.5 (2024-01-31)
-------------------

- Improve quality of coverage and format
- Fix dependency issue

v0.8.4 (2024-01-30)
-------------------

- Version cleanup

v0.8.3 (2024-01-30)
-------------------

- Initial release on PyPI

v0.8.2 (2024-01-30)
-------------------

- Improve examples
- Add module ``report`` and func ``RoutineBase.export_csv`` for results export

v0.8.1 (2024-01-20)
-------------------

- Improve ``MatProcessor``
- Add more examples
- Improve ANDES interface

v0.8.0 (2024-01-09)
-------------------

- Refactor ``DCED`` routines to improve performance

v0.7.5 (2023-12-28)
-------------------

- Refactor ``MatProcessor`` and ``DCED`` routines to improve performance
- Integrate sparsity pattern in ``RParam``
- Rename energy storage routines ``RTED2``, ``ED2`` and ``UC2`` to ``RTEDES``, ``EDES`` and ``UCES``

v0.7.4 (2023-11-29)
-------------------

- Refactor routins and optimization models to improve performance
- Fix routines modeling
- Add examples
- Fix built-in cases

v0.7.3 (2023-11-03)
-------------------

- Add tests

v0.7.2 (2023-10-26)
-------------------

- Add routines ``ED2`` and ``UC2``
- Minor fix on ``SymProcessor`` and ``Documenter``

v0.7.1 (2023-10-12)
-------------------

- Add function ``_initial_guess`` to routine ``UC``
- Refactor PYPOWER

v0.7.0 (2023-09-22)
-------------------

- Add interfaces for customizing optimization
- Add models ``REGCV1`` and ``REGCV1Cost`` for virtual inertia scheduling
- Add cost models: ``SRCost``, ``NSRCost``, ``DCost``
- Add reserve models: ``SR``, ``NSR``
- Add routine ``UC``
- Add routine ``RTED2`` to include energy storage model

v0.6.7 (2023-08-02)
-------------------

- Version cleanup

v0.6.6 (2023-07-27)
-------------------

- Improve routine reference
- Add routine ED, LDOPF

v0.6.5 (2023-06-27)
-------------------

- Update documentation with auto-generated model and routine reference
- Add interface with ANDES ``interop.andes``
- Add routine RTED and example of RTED-TDS co-simulation
- Draft development documentation

v0.6.4 (2023-05-23)
-------------------

- Setup PFlow and DCPF using PYPOWER

v0.6.3 (2023-05-22)
-------------------

- Using CVXPY for draft implementation
- Improve ``model``, ``group``, ``param`` and ``var`` in ``core``
- Refactor ``routines`` and ``opt``
- Improve PYPOWER interface ``io.pypower.system2ppc``
- Fix PYPOWER function ``solver.pypower.makePTDF``

v0.6.2 (2023-04-23)
-------------------

- Enhance docstring
- Remove unused module ``utils.LazyImport``
- Remove unused module ``shared``

v0.6.1 (2023-03-05)
-------------------

- Fix incompatiability of NumPy attribute ``object`` in  ``io.matpower._get_bus_id_caller``
- Add file parser ``io.pypower`` for PYPOWER case file
- Deprecate PYPOWER interface ``solvers.ipp``

v0.6.0 (2023-03-04)
-------------------

- Set up PYPOWER for power flow calculation
- Add PYPOWER interface ``solvers.ipp``
- Develop module ``routines`` for routine analysis
- Revise module ``system``, ``core.var``, ``core.model`` for routine analysis
- Set up routine ``PFlow`` for power flow calculation
- Add file parser ``io.matpower`` and ``io.raw`` for MATPOWER file and RAW file
- Documentation of APIs

v0.5 (2023-02-17)
-------------------

- Develop module ``system``, ``main``, ``cli``
- Development preparation: versioneer, documentation, etc.

v0.4 (2023-01)
-------------------

This release outlines the package.
