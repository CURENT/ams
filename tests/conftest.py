"""
Shared pytest fixtures for the AMS test suite.

Each fixture loads its case once per pytest session (lazy, on first
request) and yields a fresh deepcopy on every test invocation, so
tests that mutate ``ss`` (parameters, routine state, ``StaticGen.u``)
stay isolated. ``ss.reset()`` is *not* sufficient — it does not undo
parameter mutations or ``routine.initialized``.

Usage from an existing ``unittest.TestCase``::

    class TestFoo(unittest.TestCase):
        @pytest.fixture(autouse=True)
        def _attach_ss(self, pjm5bus_json):
            self.ss = pjm5bus_json

Or directly from a pytest-style function::

    def test_bar(pjm5bus_json):
        assert pjm5bus_json.DCOPF is not None

The session-scoped raw-load fixtures (``*_loaded``) are exposed for
read-only consumers that want to skip the deepcopy cost.
"""

import copy

import pytest

import ams


def _make_loaded_fixture(case_path):
    @pytest.fixture(scope="session")
    def _loaded():
        return ams.load(
            ams.get_case(case_path),
            setup=True,
            default_config=True,
            no_output=True,
        )
    return _loaded


def _make_fresh_fixture(loaded_name):
    @pytest.fixture
    def _fresh(request):
        return copy.deepcopy(request.getfixturevalue(loaded_name))
    return _fresh


pjm5bus_json_loaded = _make_loaded_fixture("5bus/pjm5bus_demo.json")
pjm5bus_xlsx_loaded = _make_loaded_fixture("5bus/pjm5bus_demo.xlsx")
case5_loaded = _make_loaded_fixture("matpower/case5.m")
case14_loaded = _make_loaded_fixture("matpower/case14.m")
ieee14_raw_loaded = _make_loaded_fixture("ieee14/ieee14.raw")
ieee39_uced_loaded = _make_loaded_fixture("ieee39/ieee39_uced.xlsx")
ieee39_uced_esd1_loaded = _make_loaded_fixture("ieee39/ieee39_uced_esd1.xlsx")


pjm5bus_json = _make_fresh_fixture("pjm5bus_json_loaded")
pjm5bus_xlsx = _make_fresh_fixture("pjm5bus_xlsx_loaded")
case5 = _make_fresh_fixture("case5_loaded")
case14 = _make_fresh_fixture("case14_loaded")
ieee14_raw = _make_fresh_fixture("ieee14_raw_loaded")
ieee39_uced = _make_fresh_fixture("ieee39_uced_loaded")
ieee39_uced_esd1 = _make_fresh_fixture("ieee39_uced_esd1_loaded")
