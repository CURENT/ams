"""
Unit commitment routines using PTDF formulations.
"""
import logging

from ams.routines.ed import ESD1MPBase, DGMPBase
from ams.routines.ed2 import PTDFMPBase
from ams.routines.uc import UC

logger = logging.getLogger(__name__)


class UC2(PTDFMPBase, UC):
    """
    DC-based unit commitment (UC) using PTDF formulations:
    The bilinear term in the formulation is linearized with big-M method.

    Non-negative var `pdu` is introduced as unserved load with its penalty `cdp`.

    Constraints include power balance, ramping, spinning reserve, non-spinning reserve,
    minimum ON/OFF duration.
    The cost inludes generation cost, startup cost, shutdown cost, spinning reserve cost,
    non-spinning reserve cost, and unserved load penalty.

    Method ``_initial_guess`` is used to make initial guess for commitment decision if all
    generators are online at initial. It is a simple heuristic method, which may not be optimal.

    Notes
    -----
    - The formulations have been adjusted with interval ``config.t``
    - The tie-line flow has not been implemented in formulations.

    References
    ----------
    1. Huang, Y., Pardalos, P. M., & Zheng, Q. P. (2017). Electrical power unit commitment: deterministic and
       two-stage stochastic programming models and algorithms. Springer.
    2. D. A. Tejada-Arango, S. Lumbreras, P. Sánchez-Martín and A. Ramos, "Which Unit-Commitment Formulation
       is Best? A Comparison Framework," in IEEE Transactions on Power Systems, vol. 35, no. 4, pp. 2926-2936,
       July 2020, doi: 10.1109/TPWRS.2019.2962024.
    """

    def __init__(self, system, config):
        super().__init__(system, config)

        # rewrite power balance to include unserved load `pdu`
        self.pb.e_str = "sum(pg, axis=0) - sum(pds - pdu, axis=0)"

        # rewrite Expression plf to include unserved load `pdu`
        self.plf.e_str = "PTDF @ (Cg@pg - Cl@(pds - pdu) - Csh@gsh@tlv - Pbusinj@tlv)"


class UC2DG(DGMPBase, UC2):
    """
    UC with distributed generation :ref:`DG`, using PTDF formulations.

    Note that UCDG only includes DG output power. If ESD1 is included,
    UCES should be used instead, otherwise there is no SOC.
    """

    def __init__(self, system, config):
        super().__init__(system, config)


class UC2ES(ESD1MPBase, UC2):
    """
    UC with energy storage :ref:`ESD1`, using PTDF formulations.
    """

    def __init__(self, system, config):
        super().__init__(system, config)
