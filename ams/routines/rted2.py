"""
RTED routines using PTDF formulation.
"""
import logging

from ams.routines.dcopf2 import PTDFBase
from ams.routines.rted import RTED, RTEDDG, RTEDESP, RTEDES


logger = logging.getLogger(__name__)


class RTED2(PTDFBase, RTED):
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
    - Formulations have been adjusted with interval ``config.t``, 5/60 [Hour] by default.
    - The tie-line flow related constraints are omitted in this formulation.
    - Power generation is balanced for the entire system.
    - SFR is balanced for each area.
    """

    def __init__(self, system, config):
        super().__init__(system, config)


class RTED2DG(PTDFBase, RTEDDG):
    """
    RTED with distributed generator :ref:`DG` using PTDF formulation.

    Note that RTED2DG only includes DG output power. If ESD1 is included,
    RTED2ES should be used instead, otherwise there is no SOC.
    """

    def __init__(self, system, config):
        super().__init__(system, config)


class RTED2ESP(PTDFBase, RTEDESP):
    """
    Price run of RTED with energy storage :ref:`ESD1` using PTDF formulation.

    This routine is not intended to work standalone. It should be used after solved
    :class:`RTED2ES`.
    When both are solved, :class:`RTED2ES` will be used.

    The binary variables ``ucd`` and ``udd`` are now parameters retrieved from
    solved :class:`RTED2ES`.

    The constraints ``zce1`` - ``zce3`` and ``zde1`` - ``zde3`` are now simplified
    to ``zce`` and ``zde`` as below:

    .. math::

        (1 - u_{cd}) * p_{ce} <= 0
        (1 - u_{dd}) * p_{de} <= 0
    """

    def __init__(self, system, config):
        super().__init__(system, config)

    def _preinit(self):
        if not self.system.RTED2ES.converged:
            raise ValueError('<RTED2ES> must be solved before <RTED2ESP>!')
        self._used_rtn = self.system.RTED2ES


class RTED2ES(PTDFBase, RTEDES):
    """
    RTED with energy storage :ref:`ESD1`.
    The bilinear term in the formulation is linearized with big-M method.

    While the formulation enforces SOCend, the ESD1 owner is not required to provide
    an SOC constraint for every RTED interval. The optimization treats SOCend
    as a terminal boundary condition, allowing the dispatcher maximum flexibility to optimize
    power output within the hour, provided the target is met at the interval's conclusion.
    """

    def __init__(self, system, config):
        super().__init__(system, config)
