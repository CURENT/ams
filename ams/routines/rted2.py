"""
RTED routines.
"""
import logging

from ams.routines.dcopf2 import PTDFMixin
from ams.routines.rted import RTED


logger = logging.getLogger(__name__)


class RTED2(PTDFMixin, RTED):
    """
    DC-based real-time economic dispatch (RTED) using PTDF.

    For large cases, it is recommended to build the PTDF first, especially when incremental
    build is necessary.

    RTED2 extends RTED with PTDF formulation:

    - Uses PTDF matrix instead of B-theta for line flow calculation
    - Calculates LMP with energy price and congestion price components
    - Inherits all RTED features: reserves, ramping, etc.

    The function ``dc2ac`` sets the ``vBus`` value from solved ACOPF.
    Without this conversion, dynamic simulation might fail due to the gap between
    DC-based dispatch results and AC-based dynamic initialization.

    Notes
    -----
    - Formulations has been adjusted with interval ``config.t``, 5/60 [Hour] by default.
    - The tie-line flow related constraints are ommited in this formulation.
    - Power generation is balanced for the entire system.
    - SFR is balanced for each area.
    """

    def __init__(self, system, config):
        # Initialize RTED base class
        RTED.__init__(self, system, config)

        # Update info to reflect RTED2's purpose
        self.info = 'Real-time economic dispatch using PTDF'
        self.type = 'DCED'

        # Setup PTDF-specific components
        self._setup_ptdf_params()
        self._setup_ptdf_expressions()

    def _post_solve(self):
        """Post-solve calculations including aBus calculation."""
        # Call parent's post_solve first
        super()._post_solve()
        # Then add PTDF-specific calculations
        self._ptdf_post_solve()
        return True

    def init(self, **kwargs):
        """Initialize the routine, checking PTDF availability."""
        self._check_ptdf()
        return super().init(**kwargs)
