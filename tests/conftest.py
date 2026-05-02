"""
Shared pytest fixtures for the AMS test suite.

Each fixture loads its case **once per worker process** (lazy, on
first request, cached in module-level state) and yields a fresh
``copy.deepcopy`` on every test invocation, so tests that mutate
``ss`` (parameters, routine state, ``StaticGen.u``) stay isolated.
``System.reset()`` is *not* sufficient — it does not undo parameter
mutations or ``routine.initialized``.

Usage from an existing ``unittest.TestCase``::

    class TestFoo(unittest.TestCase):
        @pytest.fixture(autouse=True)
        def _attach_ss(self, request, pjm5bus_json):
            request.instance.ss = pjm5bus_json

Or directly from a pytest-style function::

    def test_bar(pjm5bus_json):
        assert pjm5bus_json.DCOPF is not None

The underlying raw-loaded ``ss`` is intentionally **not** exposed as
a public fixture: a session-scoped mutable object accessed without a
deepcopy guard is order-dependent flakiness waiting to happen. If a
future read-only consumer needs to skip the deepcopy cost, add a
purpose-built fixture then — with the share-rules documented at the
call site, not here.
"""

import copy

import pytest

import ams
from ams.shared import installed_solvers, misocp_solvers


# Solver-availability flags for tests that gate per-id via
# pytest.mark.skipif. Hoisted here so the parametrized scenario
# modules under tests/routines/ share a single source of truth.
HAS_MISOCP = bool(set(misocp_solvers) & set(installed_solvers))

try:
    import pypower  # noqa: F401
    HAS_PYPOWER = True
except ImportError:
    HAS_PYPOWER = False


# Load kwargs match the original setUp() blocks across the suite:
# `setup=True` builds the System (default for ams.load),
# `default_config=True` skips ~/.ams.rc reads (test isolation),
# `no_output=True` silences write-to-disk side effects.
_LOAD_KWARGS = dict(setup=True, default_config=True, no_output=True)

_LOADED_CACHE = {}


def _load_cached(case_path):
    """Lazy per-process cache of `ams.load(case_path, **_LOAD_KWARGS)`.

    Module-level dict means each pytest worker (e.g. xdist process)
    keeps its own cache — the same lifetime as a `scope="session"`
    fixture, but without a publicly-bound name a consumer could grab
    by mistake and mutate.
    """
    if case_path not in _LOADED_CACHE:
        _LOADED_CACHE[case_path] = ams.load(
            ams.get_case(case_path),
            **_LOAD_KWARGS,
        )
    return _LOADED_CACHE[case_path]


def _make_fresh_fixture(case_path, fixture_name):
    @pytest.fixture
    def _fresh():
        return copy.deepcopy(_load_cached(case_path))

    _fresh.__name__ = fixture_name
    _fresh.__doc__ = f"Fresh deepcopy of `ams.load({case_path!r})`."
    return _fresh


pjm5bus_json = _make_fresh_fixture("5bus/pjm5bus_demo.json", "pjm5bus_json")
pjm5bus_xlsx = _make_fresh_fixture("5bus/pjm5bus_demo.xlsx", "pjm5bus_xlsx")
case5 = _make_fresh_fixture("matpower/case5.m", "case5")
case14 = _make_fresh_fixture("matpower/case14.m", "case14")
ieee14_raw = _make_fresh_fixture("ieee14/ieee14.raw", "ieee14_raw")
ieee39_uced = _make_fresh_fixture("ieee39/ieee39_uced.xlsx", "ieee39_uced")
ieee39_uced_esd1 = _make_fresh_fixture(
    "ieee39/ieee39_uced_esd1.xlsx", "ieee39_uced_esd1"
)
