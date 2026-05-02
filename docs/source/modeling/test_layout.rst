.. _test_layout:

Test suite layout
==================

The ``tests/`` tree mirrors ``ams/`` source layout: tests for code
under ``ams/<layer>/`` live under ``tests/<layer>/``. Failures point
unambiguously at the layer they exercise, and contributors can
locate the right home for a new test from the source path alone.

Tree
-----

::

    tests/
    ├── conftest.py            # shared deepcopy fixtures + solver flags
    ├── system/                # ams/system.py — addressing, repr, paths,
    │                          #   warnings, parameter correction
    ├── io/                    # ams/io/ + result-export — load, case
    │                          #   formats, csv/json export, report
    ├── core/                  # ams/core/ — non-CVXPY machinery:
    │                          #   model, routine_ns, matprocessor, service
    ├── models/                # ams/models/ — group registry
    ├── opt/                   # ams/opt/ — CVXPY wrapper layer (omodel)
    ├── routines/              # ams/routines/ — Routine base + per-family
    │                          #   parametrized scenarios
    ├── interop/               # ams/interface.py — ANDES bridge
    ├── cli/                   # ams/cli.py — argv parsing, version, dispatch
    ├── bench/                 # ams/bench/ — benchmark suite smoke
    └── codegen/               # docs/codegen + first-run pycode

Folder scope
------------

``tests/system/``
    System-level concerns that don't fit a finer layer: addressing,
    ``__repr__``, search paths, warnings, ``System.setup()``
    parameter correction.

``tests/io/``
    Case loading (matpower, psse, json, xlsx), result export
    (CSV/JSON), and report generation. The CLI's ``ams report``
    invokes the report machinery, but the report logic itself lives
    here — the CLI test only verifies the trigger (see
    ``tests/cli/`` below).

``tests/core/``
    Non-CVXPY machinery from ``ams/core/``: ``Model``, ``RoutineNS``,
    ``MatProcessor`` (PTDF/LODF/OTDF), ``Service``. These are
    infrastructure beneath the opt layer — they don't touch CVXPY.

``tests/models/``
    Device-model wiring: group registry, parameter inheritance.

``tests/opt/``
    The CVXPY wrapper layer in ``ams/opt/`` — ``OModel``,
    ``Constraint``, ``Expression``, ``Objective``. Tests that
    exercise CVXPY symbol assembly.

``tests/routines/``
    Routine subclasses and the parametrized scenario suites that
    cover the common scenario battery (init / trip_gen / trip_line /
    set_load / vBus / dc2ac) across routine families. See
    `Parametrized scenario modules`_.

``tests/interop/``
    The AMS↔ANDES bridge (``ams.interface``). ``Jumper`` device
    pass-through is folded in here because the assertion is "the
    bridge carries the device", not a Jumper-internal property.

``tests/cli/``
    The CLI entry point and the ``ams.main`` glue it dispatches to:
    argv parsing, ``--version``, the preamble, and the
    ``main.doc`` / ``main.versioninfo`` / ``main.misc`` /
    ``main.selftest`` / ``main.run(profile=True)`` paths. Subject-
    matter tests for what those paths invoke belong in the
    dedicated layer (e.g. ``ams report`` content →
    ``tests/io/test_report.py``; ``ams bench`` numbers →
    ``tests/bench/test_bench.py``). The split is *thin glue here,
    deep behaviour there*.

``tests/bench/``
    Smoke tests for ``ams/bench/`` — the benchmark suite invocation,
    not the reported numbers.

``tests/codegen/``
    Symbolic codegen and first-run pycode generation
    (``test_generator.py``), plus ``doc_all()``-style introspection
    that touches every model and group (``test_first_run_docs.py``).

Shared fixtures (``tests/conftest.py``)
----------------------------------------

The top-level ``conftest.py`` provides nine deepcopy fixtures that
every subfolder inherits via pytest's normal discovery:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Fixture
     - Loads
   * - ``pjm5bus_json``
     - ``5bus/pjm5bus_demo.json``
   * - ``pjm5bus_xlsx``
     - ``5bus/pjm5bus_demo.xlsx``
   * - ``case5``
     - ``matpower/case5.m``
   * - ``case14``
     - ``matpower/case14.m``
   * - ``case39``
     - ``matpower/case39.m``
   * - ``case118``
     - ``matpower/case118.m``
   * - ``ieee14_raw``
     - ``ieee14/ieee14.raw``
   * - ``ieee39_uced``
     - ``ieee39/ieee39_uced.xlsx``
   * - ``ieee39_uced_esd1``
     - ``ieee39/ieee39_uced_esd1.xlsx``

A tenth fixture, ``benchmark_mpres``, lazily caches
``matpower/benchmark.json`` for the known-good cross-check suite —
read-only, so no deepcopy.

Each fixture loads its case **once per worker process** (lazy
module-level cache) and yields a ``copy.deepcopy`` per test.
``System.reset()`` is *not* sufficient — it does not undo parameter
mutations or ``routine.initialized`` — so per-test isolation
requires the deepcopy.

Two solver-availability flags are also hoisted so parametrized
modules share a single source of truth:

- ``HAS_MISOCP`` — at least one of ``ams.shared.misocp_solvers`` is
  installed (gates UC/UCES/RTEDES/EDES/UC2ES/...).
- ``HAS_PYPOWER`` — ``import pypower`` succeeds (gates DCPF1,
  PFlow1, DCOPF1, ACOPF1).

Use from a ``unittest.TestCase``::

    class TestFoo(unittest.TestCase):
        @pytest.fixture(autouse=True)
        def _attach_ss(self, request, pjm5bus_json):
            request.instance.ss = pjm5bus_json

Use from a pytest-style function::

    def test_bar(case14):
        assert case14.DCOPF is not None

Parametrized scenario modules
------------------------------

Routine variants that share a common scenario body live in
parametrized modules under ``tests/routines/``. Adding a new
routine variant is a one-line ``_ROUTINES`` entry — no new class
needed.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Module
     - Family
     - Routines
   * - ``test_scenarios_lp_singleperiod.py``
     - A — single-period DC
     - DCOPF, DCOPF2, RTED, RTEDDG, RTEDES, RTED2, RTED2DG, RTED2ES
   * - ``test_scenarios_pflow.py``
     - C — power flow
     - DCPF, PFlow, DCPF1, PFlow1, DCOPF1, ACOPF1
   * - ``test_scenarios_lp_multiperiod.py``
     - D — multiperiod LP
     - ED, EDDG, EDES, ED2, ED2DG, ED2ES
   * - ``test_scenarios_milp_multiperiod.py``
     - E — multiperiod MILP
     - UC, UCDG, UCES, UC2, UC2DG, UC2ES
   * - ``test_scenarios_known_good.py``
     - MATPOWER cross-check
     - DCPF, PFlow, DCOPF, Matrices on case14/39/118

Routine-specific extras (``test_align_*``, ``test_pb_formula``,
``test_ch_decision``, ``test_export_csv``) stay in their original
``test_rtn_<name>.py`` files. Family B (ACOPF) and OPF are not
parametrized — ACOPF is a single routine; OPF has dual DC/AC modes
that don't fit a single body.

Adding a test
--------------

1. **Find the source layer.** A test for ``ams/core/<x>.py`` goes
   under ``tests/core/``. A test for a new routine variant goes
   into the matching ``tests/routines/test_scenarios_*.py`` as a
   one-line ``_ROUTINES`` entry plus any expected-values overrides.
2. **Pick a fixture.** If the case you need is already in
   ``conftest.py``, use it. If you need a new case file used in
   more than one place, add it to ``conftest.py`` via
   ``_make_fresh_fixture``.
3. **Run the local subset first.** ``pytest tests/<layer>/`` is
   fast — use it while iterating. ``pytest tests/`` runs the full
   suite (~58 s) before opening a PR.

CLI selectors
--------------

After the layer reorganization (PR #254), tests previously addressed
as ``tests/test_<name>.py::Test<Class>`` now live under
``tests/<layer>/test_<name>.py``. Pytest still discovers them
automatically; explicit selectors must be updated to include the
subfolder. The full suite count is unchanged (348 collected).
