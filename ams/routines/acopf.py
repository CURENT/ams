"""
ACOPF routines.
"""
import logging
from collections import OrderedDict

from ams.pypower import runopf
from ams.pypower.core import ppoption

from ams.io.pypower import system2ppc
from ams.core.param import RParam

from ams.routines.dcpf import DCPF
from ams.opt.omodel import Var, Constraint, Objective

logger = logging.getLogger(__name__)


class ACOPF(DCPF):
    """
    Standard AC optimal power flow.

    Notes
    -----
    1. ACOPF is solved with PYPOWER ``runopf`` function.
    2. ACOPF formulation in AMS style is NOT DONE YET,
       but this does not affect the results
       because the data are passed to PYPOWER for solving.
    """

    def __init__(self, system, config):
        DCPF.__init__(self, system, config)
        self.info = 'AC Optimal Power Flow'
        self.type = 'ACED'

        self.map1 = OrderedDict()   # ACOPF does not receive
        self.map2 = OrderedDict([
            ('Bus', {
                'vBus': 'v0',
            }),
            ('StaticGen', {
                'pg': 'p0',
            }),
        ])

        # --- params ---
        self.c2 = RParam(info='Gen cost coefficient 2',
                         name='c2', tex_name=r'c_{2}',
                         unit=r'$/(p.u.^2)', model='GCost',
                         indexer='gen', imodel='StaticGen',
                         nonneg=True)
        self.c1 = RParam(info='Gen cost coefficient 1',
                         name='c1', tex_name=r'c_{1}',
                         unit=r'$/(p.u.)', model='GCost',
                         indexer='gen', imodel='StaticGen',)
        self.c0 = RParam(info='Gen cost coefficient 0',
                         name='c0', tex_name=r'c_{0}',
                         unit=r'$', model='GCost',
                         indexer='gen', imodel='StaticGen',
                         no_parse=True)
        self.qd = RParam(info='reactive demand',
                         name='qd', tex_name=r'q_{d}',
                         model='StaticLoad', src='q0',
                         unit='p.u.',)
        # --- bus ---
        self.vBus = Var(info='Bus voltage magnitude',
                        unit='p.u.',
                        name='vBus', tex_name=r'v_{Bus}',
                        src='v', model='Bus',)
        # --- gen ---
        self.qg = Var(info='Gen reactive power',
                      unit='p.u.',
                      name='qg', tex_name=r'q_{g}',
                      model='StaticGen', src='q',)
        # --- constraints ---
        self.pb = Constraint(name='pb',
                             info='power balance',
                             e_str='sum(pd) - sum(pg)',
                             is_eq=True,)
        # TODO: ACOPF formulation
        # --- objective ---
        self.obj = Objective(name='obj',
                             info='total cost',
                             e_str='sum(c2 * pg**2 + c1 * pg + c0)',
                             sense='min',)

    def solve(self, method=None, **kwargs):
        """
        Solve ACOPF using PYPOWER with PIPS.
        """
        ppc = system2ppc(self.system)
        ppopt = ppoption()
        res, sstats = runopf(casedata=ppc, ppopt=ppopt, **kwargs)
        return res, sstats

    def run(self, **kwargs):
        """
        Run ACOPF using PYPOWER with PIPS.
        *args and **kwargs go to `self.solve()`, which are not used yet.

        Examples
        --------
        >>> ss = ams.load(ams.get_case('matpower/case14.m'))
        >>> ss.ACOPF.run()

        Parameters
        ----------
        force_init : bool
            Force initialization.
        no_code : bool
            Disable showing code.
        method : str
            Placeholder for future use.

        Returns
        -------
        exit_code : int
            Exit code of the routine.
        """
        super().run(**kwargs)
