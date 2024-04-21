"""
Module for system matrix make.
"""

import logging
from typing import Optional

import numpy as np

from scipy.sparse import csr_matrix as c_sparse
from scipy.sparse import csc_matrix as csc_sparse
from scipy.sparse import lil_matrix as l_sparse

from andes.utils.misc import elapsed

from ams.opt.omodel import Param

logger = logging.getLogger(__name__)


class MParam(Param):
    """
    Class for matrix parameters built from the system.

    MParam is designed to be a subclass of RParam for routine parameters
    management.

    Parameters
    ----------
    name : str, optional
        Name of this parameter. If not provided, `name` will be set
        to the attribute name.
    tex_name : str, optional
        LaTeX-formatted parameter name. If not provided, `tex_name`
        will be assigned the same as `name`.
    info : str, optional
        A description of this parameter
    unit : str, optional
        Unit of the parameter.
    v : np.ndarray, optional
        Matrix value of the parameter.
    owner : object, optional
        Owner of the MParam, usually the MatProcessor instance.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 v: Optional[np.ndarray] = None,
                 sparse: Optional[bool] = False,
                 ):
        self.name = name
        self.tex_name = tex_name if (tex_name is not None) else name
        self.info = info
        self.unit = unit
        self._v = v
        self.sparse = sparse
        self.owner = None

    @property
    def v(self):
        """
        Return the value of the parameter.
        """
        # NOTE: scipy.sparse matrix will return 2D array
        # so we squeeze it here if only one row
        if isinstance(self._v, (c_sparse, l_sparse, csc_sparse)):
            out = self._v.toarray()
            if out.shape[0] == 1:
                return np.squeeze(out)
            else:
                return out
        return self._v

    @property
    def shape(self):
        """
        Return the shape of the parameter.
        """
        return self.v.shape

    @property
    def n(self):
        """
        Return the szie of the parameter.
        """
        return len(self.v)

    @property
    def class_name(self):
        """
        Return the class name
        """
        return self.__class__.__name__


class MatProcessor:
    """
    Class for matrix processing in AMS system.
    """

    def __init__(self, system):
        self.system = system
        self.initialized = False

        self.Cft = MParam(name='Cft', tex_name=r'C_{ft}',
                          info='Line connectivity matrix',
                          v=None, sparse=True)
        self.CftT = MParam(name='CftT', tex_name=r'C_{ft}^{T}',
                           info='Line connectivity matrix transpose',
                           v=None, sparse=True)
        self.Cg = MParam(name='Cg', tex_name=r'C_g',
                         info='Generator connectivity matrix',
                         v=None, sparse=True)
        self.Cl = MParam(name='Cl', tex_name=r'Cl',
                         info='Load connectivity matrix',
                         v=None, sparse=True)
        self.Csh = MParam(name='Csh', tex_name=r'C_{sh}',
                          info='Shunt connectivity matrix',
                          v=None, sparse=True)

        self.Bbus = MParam(name='Bbus', tex_name=r'B_{bus}',
                           info='Bus admittance matrix',
                           v=None, sparse=True)
        self.Bf = MParam(name='Bf', tex_name=r'B_{f}',
                         info='Bf matrix',
                         v=None, sparse=True)
        self.Pbusinj = MParam(name='Pbusinj', tex_name=r'P_{bus}^{inj}',
                              info='Bus power injection vector',
                              v=None,)
        self.Pfinj = MParam(name='Pfinj', tex_name=r'P_{f}^{inj}',
                            info='Line power injection vector',
                            v=None,)

        self.PTDF = MParam(name='PTDF', tex_name=r'P_{TDF}',
                           info='Power transfer distribution factor',
                           v=None)
        self.LODF = MParam(name='LODF', tex_name=r'O_{TDF}',
                           info='Line outage distribution factor',
                           v=None)

    def build(self):
        """
        Build the system matrices.
        It build connectivity matrices first: Cg, Cl, Csh, Cft, and CftT.
        Then build bus matrices: Bf, Bbus, Pfinj, and Pbusinj.

        Returns
        -------
        initialized : bool
            True if the matrices are built successfully.
        """
        t_mat, _ = elapsed()

        # --- connectivity matrices ---
        _ = self.build_cg()
        _ = self.build_cl()
        _ = self.build_csh()
        _ = self.build_cft()

        # --- bus matrices ---
        _ = self.build_bf()
        _ = self.build_bbus()
        _ = self.build_pfinj()
        _ = self.build_pbusinj()
        _, s_mat = elapsed(t_mat)

        logger.debug(f"Built system matrices in {s_mat}.")
        self.initialized = True
        return self.initialized

    @property
    def class_name(self):
        """
        Return the class name
        """
        return self.__class__.__name__

    @property
    def n(self):
        """
        To fit the RParam style.
        """
        return 2

    def build_cg(self):
        """
        Build generator connectivity matrix Cg, and store it in the MParam `Cg`.

        Returns
        -------
        Cg : scipy.sparse.csr_matrix
            Generator connectivity matrix.
        """
        system = self.system

        # common variables
        nb = system.Bus.n
        ng = system.StaticGen.n

        # bus indices: idx -> uid
        idx_gen = system.StaticGen.get_idx()
        u_gen = system.StaticGen.get(src='u', attr='v', idx=idx_gen)
        on_gen = np.flatnonzero(u_gen)  # uid of online generators
        on_gen_idx = [idx_gen[i] for i in on_gen]  # idx of online generators
        on_gen_bus = system.StaticGen.get(src='bus', attr='v', idx=on_gen_idx)

        row = np.array([system.Bus.idx2uid(x) for x in on_gen_bus])
        col = np.array([idx_gen.index(x) for x in on_gen_idx])
        self.Cg._v = c_sparse((np.ones(len(on_gen_idx)), (row, col)), (nb, ng))
        return self.Cg._v

    def build_cl(self):
        """
        Build load connectivity matrix Cl, and store it in the MParam `Cl`.

        Returns
        -------
        Cl : scipy.sparse.csr_matrix
            Load connectivity matrix.
        """
        system = self.system

        # common variables
        nb = system.Bus.n
        npq = system.PQ.n

        # load indices: idx -> uid
        idx_load = system.PQ.idx.v
        u_load = system.PQ.get(src='u', attr='v', idx=idx_load)
        on_load = np.flatnonzero(u_load)
        on_load_idx = [idx_load[i] for i in on_load]
        on_load_bus = system.PQ.get(src='bus', attr='v', idx=on_load_idx)

        row = np.array([system.Bus.idx2uid(x) for x in on_load_bus])
        col = np.array([system.PQ.idx2uid(x) for x in on_load_idx])
        self.Cl._v = c_sparse((np.ones(len(on_load_idx)), (row, col)), (nb, npq))
        return self.Cl._v

    def build_csh(self):
        """
        Build shunt connectivity matrix Csh, and store it in the MParam `Csh`.

        Returns
        -------
        Csh : spmatrix
            Shunt connectivity matrix.
        """
        system = self.system

        # common variables
        nb = system.Bus.n
        nsh = system.Shunt.n

        # shunt indices: idx -> uid
        idx_shunt = system.Shunt.idx.v
        u_shunt = system.Shunt.get(src='u', attr='v', idx=idx_shunt)
        on_shunt = np.flatnonzero(u_shunt)
        on_shunt_idx = [idx_shunt[i] for i in on_shunt]
        on_shunt_bus = system.Shunt.get(src='bus', attr='v', idx=on_shunt_idx)

        row = np.array([system.Bus.idx2uid(x) for x in on_shunt_bus])
        col = np.array([system.Shunt.idx2uid(x) for x in on_shunt_idx])
        self.Csh._v = c_sparse((np.ones(len(on_shunt_idx)), (row, col)), (nb, nsh))
        return self.Csh._v

    def build_cft(self):
        """
        Build line connectivity matrix Cft and its transpose CftT.
        The Cft and CftT are stored in the MParam `Cft` and `CftT`, respectively.

        Returns
        -------
        Cft : scipy.sparse.csr_matrix
            Line connectivity matrix.
        """
        system = self.system

        # common variables
        nb = system.Bus.n
        nl = system.Line.n

        # line indices: idx -> uid
        idx_line = system.Line.idx.v
        u_line = system.Line.get(src='u', attr='v', idx=idx_line)
        on_line = np.flatnonzero(u_line)
        on_line_idx = [idx_line[i] for i in on_line]
        on_line_bus1 = system.Line.get(src='bus1', attr='v', idx=on_line_idx)
        on_line_bus2 = system.Line.get(src='bus2', attr='v', idx=on_line_idx)

        data_line = np.ones(2*len(on_line_idx))
        data_line[len(on_line_idx):] = -1
        row_line = np.array([system.Bus.idx2uid(x) for x in on_line_bus1 + on_line_bus2])
        col_line = np.array([system.Line.idx2uid(x) for x in on_line_idx + on_line_idx])
        self.Cft._v = c_sparse((data_line, (row_line, col_line)), (nb, nl))
        self.CftT._v = self.Cft._v.T
        return self.Cft._v

    def build_bf(self):
        """
        Build DC Bf matrix and store it in the MParam `Bf`.

        Returns
        -------
        Bf : scipy.sparse.csr_matrix
            Bf matrix.
        """
        system = self.system

        # common variables
        nb = system.Bus.n
        nl = system.Line.n

        # line parameters
        idx_line = system.Line.idx.v
        b = self._calc_b()

        # build Bf such that Bf * Va is the vector of real branch powers injected
        # at each branch's "from" bus
        f = system.Bus.idx2uid(system.Line.get(src='bus1', attr='v', idx=idx_line))
        t = system.Bus.idx2uid(system.Line.get(src='bus2', attr='v', idx=idx_line))
        ir = np.r_[range(nl), range(nl)]  # double set of row indices
        self.Bf._v = c_sparse((np.r_[b, -b], (ir, np.r_[f, t])), (nl, nb))
        return self.Bf._v

    def build_bbus(self):
        """
        Build Bdc matrix and store it in the MParam `Bbus`.

        Returns
        -------
        Bdc : scipy.sparse.csr_matrix
            DC bus admittance matrix.
        """
        self.Bbus._v = self.Cft._v * self.Bf._v
        return self.Bbus._v

    def build_pfinj(self):
        """
        Build DC Pfinj vector and store it in the MParam `Pfinj`.

        Returns
        -------
        Pfinj : np.ndarray
            Line power injection vector.
        """
        idx_line = self.system.Line.idx.v
        b = self._calc_b()
        phi = self.system.Line.get(src='phi', attr='v', idx=idx_line)
        self.Pfinj._v = b * (-phi)
        return self.Pfinj._v

    def build_pbusinj(self):
        """
        Build DC Pbusinj vector and store it in the MParam `Pbusinj`.

        Returns
        -------
        Pbusinj : np.ndarray
            Bus power injection vector.
        """
        self.Pbusinj._v = self.Cft._v * self.Pfinj._v
        return self.Pbusinj._v

    def _calc_b(self):
        """
        Calculate DC series susceptance for each line.

        Returns
        -------
        b : np.ndarray
            Series susceptance for each line.
        """
        system = self.system
        nl = system.Line.n

        # line parameters
        idx_line = system.Line.idx.v
        x = system.Line.get(src='x', attr='v', idx=idx_line)
        u_line = system.Line.get(src='u', attr='v', idx=idx_line)
        b = u_line / x  # series susceptance

        # in DC, tap is assumed to be 1
        tap0 = system.Line.get(src='tap', attr='v', idx=idx_line)
        tap = np.ones(nl)
        i = np.flatnonzero(tap0)
        tap[i] = tap0[i]  # assign non-zero tap ratios
        b = b / tap  # adjusted series susceptance

        return b

    def build_ptdf(self):
        """
        Build the DC PTDF matrix and store it in the MParam `PTDF`.

        Note that there is discrepency between the PTDF-based line flow and
        DCOPF calcualted line flow. The gap is ignorable for small cases.

        Returns
        -------
        PTDF : np.ndarray
            Power transfer distribution factor.
        """
        system = self.system

        # common variables
        nb = system.Bus.n
        nl = system.Line.n

        # use first slack bus as reference slack bus
        slack = system.Slack.bus.v[0]

        # use first bus for voltage angle reference
        noref_idx = system.Bus.idx.v[1:]
        noref = system.Bus.idx2uid(noref_idx)

        noslack = [system.Bus.idx2uid(bus) for bus in system.Bus.idx.v if bus != slack]

        # build other matrices if not built
        if not self.initialized:
            logger.debug("System matrices are not built. Building now.")
            self.build()
        # use dense representation
        Bbus, Bf = self.Bbus.v, self.Bf.v

        # initialize PTDF matrix
        H = np.zeros((nl, nb))
        # calculate PTDF
        H[:, noslack] = np.linalg.solve(Bbus[np.ix_(noslack, noref)].T, Bf[:, noref].T).T
        # store PTDF
        self.PTDF._v = H

        return self.PTDF._v

    def build_lodf(self):
        """
        Build the DC LODF matrix and store it in the MParam `LODF`.

        `LODF[m, n]` means the increased line flow on line `m` when there is
        1 p.u. line flow decrease on line `n` due to line `n` outage.

        Returns
        -------
        LODF : np.ndarray
            Line outage distribution factor.
        """
        system = self.system

        # common variables
        nl = system.Line.n

        # build PTDF if not built
        if self.PTDF._v is None:
            self.build_ptdf()

        H = self.PTDF._v * self.Cft._v
        h = np.diag(H, 0)
        LODF = H / (np.ones((nl, nl)) - np.ones((nl, 1)) * h.T)
        LODF = LODF - np.diag(np.diag(LODF)) - np.eye(nl, nl)

        self.LODF._v = LODF
        return self.LODF._v
