"""
Tests for the v1.3.0 :class:`ams.core.param.RParam` ``horizon=`` /
``hindexer=`` 2D-pivot path. The integration scenarios in
``tests/routines/`` exercise the happy path end-to-end; these unit
tests cover the contract assertions and error-message branches that
the integration tests don't reach.
"""

import unittest

import numpy as np
import pytest

import ams
from ams.core.param import RParam


def _build_minimal_horizon_system():
    """
    Construct a tiny system with two areas, two slots, and partial
    EDSlotLoad rows so the default-fill path is exercised.
    """
    ss = ams.System()
    ss.add('Area', dict(idx='A1', u=1.0, name='A1'))
    ss.add('Area', dict(idx='A2', u=1.0, name='A2'))
    ss.add('EDSlot', dict(idx='S1', u=1.0, name='S1'))
    ss.add('EDSlot', dict(idx='S2', u=1.0, name='S2'))
    # (A2, S2) intentionally omitted — should default-fill to 1.0.
    ss.add('EDSlotLoad', dict(area='A1', slot='S1', sd=0.5))
    ss.add('EDSlotLoad', dict(area='A1', slot='S2', sd=0.7))
    ss.add('EDSlotLoad', dict(area='A2', slot='S1', sd=1.1))
    ss.setup()
    return ss


def _wire_test_rparams(ss):
    """
    Stand up a ``timeslot`` + ``sd`` RParam pair without spinning up
    a full routine — fakes the routine handle the RParam needs.
    """
    class FakeRtn:
        pass
    fake = FakeRtn()
    fake.system = ss

    timeslot = RParam(name='timeslot', src='idx', model='EDSlot',
                      no_parse=True)
    timeslot.rtn = fake
    timeslot.owner = ss.EDSlot

    sd = RParam(name='sd', src='sd', model='EDSlotLoad',
                indexer='area', imodel='Area',
                horizon=timeslot, hindexer='slot')
    sd.rtn = fake
    sd.owner = ss.EDSlotLoad
    return timeslot, sd


class TestRParamHorizon(unittest.TestCase):

    def test_2d_pivot_shape_and_ordering(self):
        """The pivot returns ``(primary.n, secondary.n)`` ordered by
        ``imodel.get_all_idxes()`` × ``horizon.v``."""
        ss = _build_minimal_horizon_system()
        _, sd = _wire_test_rparams(ss)

        result = sd.v
        self.assertEqual(result.shape, (2, 2))
        # Row 0 (A1): S1=0.5, S2=0.7. Row 1 (A2): S1=1.1, S2=default 1.0.
        np.testing.assert_array_equal(result, np.array([[0.5, 0.7],
                                                        [1.1, 1.0]]))

    def test_default_fill_uses_numparam_default(self):
        """Cells with no matching row default to the source
        :class:`NumParam.default`."""
        ss = _build_minimal_horizon_system()
        _, sd = _wire_test_rparams(ss)

        # EDSlotLoad.sd has default=1.0; (A2, S2) was omitted.
        result = sd.v
        self.assertEqual(result[1, 1], 1.0)

    def test_duplicate_rows_raise(self):
        """Two rows with the same ``(primary, secondary)`` pair raise
        with the offending key in the error."""
        ss = ams.System()
        ss.add('Area', dict(idx='A1', u=1.0, name='A1'))
        ss.add('EDSlot', dict(idx='S1', u=1.0, name='S1'))
        ss.add('EDSlotLoad', dict(area='A1', slot='S1', sd=0.5))
        ss.add('EDSlotLoad', dict(area='A1', slot='S1', sd=0.7))
        ss.setup()
        _, sd = _wire_test_rparams(ss)

        with pytest.raises(ValueError, match="duplicate row"):
            _ = sd.v

    def test_missing_hindexer_construct_time_raises(self):
        """``horizon=`` without ``hindexer=`` raises at construction."""
        ss = _build_minimal_horizon_system()
        timeslot, _ = _wire_test_rparams(ss)
        with pytest.raises(ValueError, match="hindexer"):
            RParam(name='bad', src='sd', model='EDSlotLoad',
                   indexer='area', imodel='Area', horizon=timeslot)

    def test_missing_hindexer_late_bind_raises(self):
        """Late-bound ``horizon`` without ``hindexer`` raises at
        materialize time (defense-in-depth)."""
        ss = _build_minimal_horizon_system()
        timeslot, _ = _wire_test_rparams(ss)

        bad = RParam(name='bad', src='sd', model='EDSlotLoad',
                     indexer='area', imodel='Area')
        bad.rtn = timeslot.rtn
        bad.owner = ss.EDSlotLoad
        bad.horizon = timeslot   # late-bind without hindexer
        with pytest.raises(ValueError, match="hindexer"):
            _ = bad.v

    def test_unresolved_imodel_raises(self):
        """An unknown ``imodel`` raises with a clear message."""
        ss = _build_minimal_horizon_system()
        timeslot, sd = _wire_test_rparams(ss)
        sd.imodel = 'NoSuchModel'
        with pytest.raises(AttributeError, match="primary indexer"):
            _ = sd.v


class TestRParamShape(unittest.TestCase):

    def test_shape_reports_2d_for_late_bound_no_parse(self):
        """An RParam late-bound to a 2D source still reports a real
        ``shape`` even if ``no_parse=True`` (regression guard for
        the inherited DCPF.ug case)."""
        ss = _build_minimal_horizon_system()
        timeslot, _ = _wire_test_rparams(ss)
        # Force the no_parse=True flag like the inherited DCPF.ug.
        sd = RParam(name='sd', src='sd', model='EDSlotLoad',
                    indexer='area', imodel='Area',
                    horizon=timeslot, hindexer='slot',
                    no_parse=True)
        sd.rtn = timeslot.rtn
        sd.owner = ss.EDSlotLoad
        self.assertEqual(sd.shape, (2, 2))
