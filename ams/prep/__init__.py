"""
AOT codegen for AMS routines (Phase 4.5+).

Walks a fully-constructed routine instance and emits a Python module
of named callables — one per ``Constraint``/``Expression``/``Objective``/
``ExpressionCalc`` with an ``e_str``. The generated functions take a
:class:`ams.core.routine_ns.RoutineNS` and return the same shape
``e_fn=...`` already accepts, so the routine can wire them in at init
time without any regex/eval at runtime.

Authors keep writing ``e_str`` strings — the terse DSL is preserved.
``ams prep`` (or auto-prep on first init) is what produces the runtime
artefacts.

Public API:

- :func:`generate_for_routine` — emit source for one routine.
- :func:`prep_all` — emit source for every registered routine class.
"""

from ams.prep.generator import generate_for_routine, source_md5  # noqa: F401
