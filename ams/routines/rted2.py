"""
RTED routines using PTDF formulation.
"""
import logging

from ams.routines.dcopf2 import PTDFMixin
from ams.routines.rted import RTED, DGBase, ESD1Base


logger = logging.getLogger(__name__)


class RTED2(RTED, PTDFMixin):
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
        RTED.__init__(self, system, config)
        PTDFMixin.__init__(self)


class RTED2DG(RTED2, DGBase):
    """
    RTED2 with distributed generator :ref:`DG`.

    Note that RTED2DG only includes DG output power. If ESD1 is included,
    RTED2ES should be used instead, otherwise there is no SOC.
    """

    def __init__(self, system, config):
        RTED2.__init__(self, system, config)
        DGBase.__init__(self)


class RTED2ES(RTED2, ESD1Base):
    """
    RTED with energy storage :ref:`ESD1`.
    The bilinear term in the formulation is linearized with big-M method.

    While the formulation enforces SOCend, the ESD1 owner is not required to provide
    an SOC constraint for every 5-minute RTED interval. The optimization treats SOCend
    as a terminal boundary condition, allowing the dispatcher maximum flexibility to optimize
    power output within the hour, provided the target is met at the interval's conclusion.
    """

    def __init__(self, system, config):
        RTED2.__init__(self, system, config)
        ESD1Base.__init__(self)
