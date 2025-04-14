"""
Module for system matrix make.
"""

import logging
import os
import sys
from typing import Optional

import numpy as np

from andes.thirdparty.npfunc import safe_div
from andes.shared import tqdm, tqdm_nb
from andes.utils.misc import elapsed, is_notebook

from ams.opt import Param
from ams.shared import pd, sps

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
    sparse : bool, optional
        If True, the matrix is stored in sparse format.
    col_names : list, optional
        Column names of the matrix.
    row_names : list, optional
        Row names of the matrix.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 v: Optional[np.ndarray] = None,
                 owner: Optional[object] = None,
                 sparse: Optional[bool] = False,
                 col_names: Optional[list] = None,
                 row_names: Optional[list] = None,
                 ):
        self.name = name
        self.tex_name = tex_name if (tex_name is not None) else name
        self.info = info
        self.unit = unit
        self._v = v
        self.sparse = sparse
        self.owner = owner
        self.col_names = col_names
        self.row_names = row_names

    def export_csv(self, path=None):
        """
        Export the matrix to a CSV file.

        Parameters
        ----------
        path : str, optional
            Path to the output CSV file.

        Returns
        -------
        str
            The path of the exported csv file
        """

        if path is None:
            if self.owner.system.files.fullname is None:
                logger.info("Input file name not detacted. Using `Untitled`.")
                file_name = f'Untitled_{self.name}'
            else:
                file_name = os.path.splitext(self.owner.system.files.fullname)[0]
                file_name += f'_{self.name}'
            path = os.path.join(os.getcwd(), file_name + '.csv')
        else:
            file_name = os.path.splitext(os.path.basename(path))[0]

        pd.DataFrame(data=self.v, columns=self.col_names, index=self.row_names).to_csv(path)

        return file_name + '.csv'

    @property
    def v(self):
        """
        Return the value of the parameter.
        """
        # NOTE: scipy.sparse matrix will return 2D array
        # so we squeeze it here if only one row
        if isinstance(self._v, (sps.csr_matrix, sps.lil_matrix, sps.csc_matrix)):
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
    Class for matrices processing in AMS system.
    The connectivity matrices `Cft`, `Cg`, `Cl`, and `Csh` ***have taken*** the
    devices connectivity into account.

    The MParams' row names and col names are assigned in `System.setup()`.
    """

    def __init__(self, system):
        self.system = system
        self.initialized = False
        self.pbar = None

        self.Cft = MParam(name='Cft', tex_name=r'C_{ft}',
                          info='Line connectivity matrix',
                          v=None, sparse=True, owner=self)
        self.CftT = MParam(name='CftT', tex_name=r'C_{ft}^{T}',
                           info='Line connectivity matrix transpose',
                           v=None, sparse=True, owner=self)
        self.Cg = MParam(name='Cg', tex_name=r'C_g',
                         info='Generator connectivity matrix',
                         v=None, sparse=True, owner=self)
        self.Cl = MParam(name='Cl', tex_name=r'Cl',
                         info='Load connectivity matrix',
                         v=None, sparse=True, owner=self)
        self.Csh = MParam(name='Csh', tex_name=r'C_{sh}',
                          info='Shunt connectivity matrix',
                          v=None, sparse=True, owner=self)

        self.Bbus = MParam(name='Bbus', tex_name=r'B_{bus}',
                           info='Bus admittance matrix',
                           v=None, sparse=True, owner=self)
        self.Bf = MParam(name='Bf', tex_name=r'B_{f}',
                         info='Bf matrix',
                         v=None, sparse=True, owner=self)
        self.Pbusinj = MParam(name='Pbusinj', tex_name=r'P_{bus}^{inj}',
                              info='Bus power injection vector',
                              v=None, sparse=False, owner=self)
        self.Pfinj = MParam(name='Pfinj', tex_name=r'P_{f}^{inj}',
                            info='Line power injection vector',
                            v=None, sparse=False, owner=self)

        self.PTDF = MParam(name='PTDF', tex_name=r'P_{TDF}',
                           info='Power transfer distribution factor',
                           v=None, sparse=False, owner=self)
        self.LODF = MParam(name='LODF', tex_name=r'O_{TDF}',
                           info='Line outage distribution factor',
                           v=None, sparse=False, owner=self)

    def build(self, force=False):
        """
        Build the system matrices.
        It build connectivity matrices first: Cg, Cl, Csh, Cft, and CftT.
        Then build bus matrices: Bf, Bbus, Pfinj, and Pbusinj.

        Parameters
        ----------
        force : bool, optional
            If True, force to rebuild the matrices. Default is False.

        Notes
        -----
        Generator online status is NOT considered in its connectivity matrix.
        The same applies for load, line, and shunt.

        Returns
        -------
        initialized : bool
            True if the matrices are built successfully.
        """
        if not force and self.initialized:
            logger.debug("System matrices are already built.")
            return self.initialized

        t_mat, _ = elapsed()
        logger.warning("Building system matrices")
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

        logger.debug(f" -> System matrices built in {s_mat}")
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
        idx_gen = system.StaticGen.get_all_idxes()
        u_gen = system.StaticGen.get(src='u', attr='v', idx=idx_gen)
        on_gen = np.flatnonzero(u_gen)  # uid of online generators
        on_gen_idx = [idx_gen[i] for i in on_gen]  # idx of online generators
        on_gen_bus = system.StaticGen.get(src='bus', attr='v', idx=on_gen_idx)

        row = np.array([system.Bus.idx2uid(x) for x in on_gen_bus])
        col = np.array([idx_gen.index(x) for x in on_gen_idx])
        self.Cg._v = sps.csr_matrix((np.ones(len(on_gen_idx)), (row, col)), (nb, ng))
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
        self.Cl._v = sps.csr_matrix((np.ones(len(on_load_idx)), (row, col)), (nb, npq))
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
        self.Csh._v = sps.csr_matrix((np.ones(len(on_shunt_idx)), (row, col)), (nb, nsh))
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
        self.Cft._v = sps.csr_matrix((data_line, (row_line, col_line)), (nb, nl))
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
        self.Bf._v = sps.csr_matrix((np.r_[b, -b], (ir, np.r_[f, t])), (nl, nb))
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

    def build_ptdf(self, line=None, no_store=False,
                   incremental=False, step=1000, no_tqdm=False,
                   permc_spec=None, use_umfpack=True):
        """
        Build the Power Transfer Distribution Factor (PTDF) matrix and optionally store it in `MParam.PTDF`.

        PTDF[m, n] represents the increased line flow on line `m` for a 1 p.u. power injection at bus `n`.
        It is similar to the Generation Shift Factor (GSF).

        Note: There may be minor discrepancies between PTDF-based line flow and DCOPF-calculated line flow.

        For large cases, use `incremental=True` to calculate the sparse PTDF in chunks, which will be stored
        as a `scipy.sparse.lil_matrix`. In this mode, the PTDF is calculated in chunks, and a progress bar
        will be shown unless `no_tqdm=True`.

        Parameters
        ----------
        line : int, str, list, optional
            Line indices for which the PTDF is calculated. If specified, the PTDF will not be stored in `MParam`.
        no_store : bool, optional
            If False, store the PTDF in `MatProcessor.PTDF._v`. Default is False.
        incremental : bool, optional
            If True, calculate the sparse PTDF in chunks to save memory. Default is False.
        step : int, optional
            Step size for incremental calculation. Default is 1000.
        no_tqdm : bool, optional
            If True, disable the progress bar. Default is False.
        permc_spec : str, optional
            Column permutation strategy for sparsity preservation. Default is 'COLAMD'.
        use_umfpack : bool, optional
            If True, use UMFPACK as the solver. Effective only when (`incremental=True`)
            & (`line` contains a single line or `step` is 1). Default is True.

        Returns
        -------
        PTDF : np.ndarray or scipy.sparse.lil_matrix
            Power transfer distribution factor.

        References
        ----------
        1. PowerWorld Documentation, Power Transfer Distribution Factors,
           https://www.powerworld.com/WebHelp/Content/MainDocumentation_HTML/Power_Transfer_Distribution_Factors.htm
        """
        system = self.system

        # use first slack bus as reference slack bus
        slack = system.Slack.bus.v[0]
        noslack = [system.Bus.idx2uid(bus) for bus in system.Bus.idx.v if bus != slack]

        # use first bus for voltage angle reference
        noref_idx = system.Bus.idx.v[1:]
        noref = system.Bus.idx2uid(noref_idx)

        if line is None:
            luid = system.Line.idx2uid(system.Line.idx.v)
        elif isinstance(line, (int, str)):
            try:
                luid = [system.Line.idx2uid(line)]
            except ValueError:
                raise ValueError(f"Line {line} not found.")
        elif isinstance(line, list):
            luid = system.Line.idx2uid(line)

        # build other matrices if not built
        if not self.initialized:
            logger.debug("System matrices are not built. Building now.")
            self.build()

        nbus = system.Bus.n
        nline = len(luid)

        Bbus = self.Bbus._v
        Bf = self.Bf._v

        if incremental:
            # initialize progress bar
            if is_notebook():
                self.pbar = tqdm_nb(total=100, unit='%', file=sys.stdout,
                                    disable=no_tqdm)
            else:
                self.pbar = tqdm(total=100, unit='%', ncols=80, ascii=True,
                                 file=sys.stdout, disable=no_tqdm)

            self.pbar.update(0)
            last_pc = 0

            H = sps.lil_matrix((nline, system.Bus.n))

            # NOTE: for PTDF, we are building rows by rows
            for start in range(0, nline, step):
                end = min(start + step, nline)
                sol = sps.linalg.spsolve(A=Bbus[np.ix_(noslack, noref)].T,
                                         b=Bf[np.ix_(luid[start:end], noref)].T,
                                         permc_spec=permc_spec,
                                         use_umfpack=use_umfpack).T
                H[start:end, noslack] = sol

                # show progress in percentage
                perc = np.round(min((end / nline) * 100, 100), 2)

                perc_diff = perc - last_pc
                if perc_diff >= 1:
                    self.pbar.update(perc_diff)
                    last_pc = perc

            # finish progress bar
            self.pbar.update(100 - last_pc)
            # removed `pbar` so that System object can be serialized
            self.pbar.close()
            self.pbar = None
        else:
            H = np.zeros((nline, nbus))
            H[:, noslack] = np.linalg.solve(Bbus.todense()[np.ix_(noslack, noref)].T,
                                            Bf.todense()[np.ix_(luid, noref)].T).T

        # reshape results into 1D array if only one line
        if isinstance(line, (int, str)):
            H = H[0, :]

        if (not no_store) & (line is None):
            self.PTDF._v = H

        return H

    def build_lodf(self, line=None, no_store=False,
                   incremental=False, step=1000, no_tqdm=False):
        """
        Build the Line Outage Distribution Factor matrix and store it in the
        MParam `LODF`.

        `LODF[m, n]` means the increased line flow on line `m` when there is
        1 p.u. line flow decrease on line `n` due to line `n` outage.
        It is also referred to as Branch Outage Distribution Factor (BODF).

        It requires DC PTDF and Cft.

        For large cases where memory is a concern, use `incremental=True` to
        calculate the sparse LODF in chunks in the format of scipy.sparse.lil_matrix.

        Parameters
        ----------
        line: int, str, list, optional
            Lines index for which the LODF is calculated. It takes both single
            or multiple line indices. Note that if `line` is given, the LODF will
            not be stored in the MParam.
        no_store : bool, optional
            If False, the LODF will be stored into `MatProcessor.LODF._v`.
        incremental : bool, optional
            If True, the sparse LODF will be calculated in chunks to save memory.
        step : int, optional
            Step for incremental calculation.
        no_tqdm : bool, optional
            If True, the progress bar will be disabled.

        Returns
        -------
        LODF : np.ndarray, scipy.sparse.lil_matrix
            Line outage distribution factor.

        References
        ----------
        1. PowerWorld Documentation, Line Outage Distribution Factors,
           https://www.powerworld.com/WebHelp/Content/MainDocumentation_HTML/Line_Outage_Distribution_Factors_LODFs.htm
        """
        system = self.system

        if line is None:
            luid = system.Line.idx2uid(system.Line.idx.v)
        elif isinstance(line, (int, str)):
            try:
                luid = [system.Line.idx2uid(line)]
            except ValueError:
                raise ValueError(f"Line {line} not found.")
        elif isinstance(line, list):
            luid = system.Line.idx2uid(line)

        # NOTE: here we use nbranch to differentiate it with nline
        nbranch = system.Line.n
        nline = len(luid)

        ptdf = self.PTDF._v
        # build PTDF if not built
        if self.PTDF._v is None:
            ptdf = self.build_ptdf(no_store=True, incremental=incremental, step=step)
        if incremental and isinstance(self.PTDF._v, np.ndarray):
            ptdf = sps.lil_matrix(self.PTDF._v)

        if incremental | (isinstance(ptdf, sps.spmatrix)):
            # initialize progress bar
            if is_notebook():
                self.pbar = tqdm_nb(total=100, unit='%', file=sys.stdout,
                                    disable=no_tqdm)
            else:
                self.pbar = tqdm(total=100, unit='%', ncols=80, ascii=True,
                                 file=sys.stdout, disable=no_tqdm)

            self.pbar.update(0)
            last_pc = 0

            LODF = sps.lil_matrix((nbranch, nline))

            # NOTE: for LODF, we are doing it columns by columns
            # reshape luid to list of list by step
            luidp = [luid[i:i + step] for i in range(0, len(luid), step)]
            for luidi in luidp:
                H_chunk = ptdf @ self.Cft._v[:, luidi]
                h_chunk = H_chunk.diagonal(-luidi[0])
                rden = safe_div(np.ones(H_chunk.shape),
                                np.tile(np.ones_like(h_chunk) - h_chunk, (nbranch, 1)))
                H_chunk = H_chunk.multiply(rden).tolil()
                # NOTE: use lil_matrix to set diagonal values as -1
                rsid = sps.diags(H_chunk.diagonal(-luidi[0])) + sps.eye(H_chunk.shape[1])
                if H_chunk.shape[0] > rsid.shape[0]:
                    Rsid = sps.lil_matrix(H_chunk.shape)
                    Rsid[luidi, :] = rsid
                else:
                    Rsid = rsid
                H_chunk = H_chunk - Rsid
                LODF[:, [luid.index(i) for i in luidi]] = H_chunk

                # show progress in percentage
                perc = np.round(min((luid.index(luidi[-1]) / nline) * 100, 100), 2)

                perc_diff = perc - last_pc
                if perc_diff >= 1:
                    self.pbar.update(perc_diff)
                    last_pc = perc

            # finish progress bar
            self.pbar.update(100 - last_pc)
            # removed `pbar` so that System object can be serialized
            self.pbar.close()
            self.pbar = None
        else:
            H = ptdf @ self.Cft._v[:, luid]
            h = np.diag(H, -luid[0])
            LODF = safe_div(H,
                            np.tile(np.ones_like(h) - h, (nbranch, 1)))
            # # NOTE: reset the diagonal elements to -1
            rsid = np.diag(np.diag(LODF, -luid[0])) + np.eye(nline, nline)
            if LODF.shape[0] > rsid.shape[0]:
                Rsid = np.zeros_like(LODF)
                Rsid[luid, :] = rsid
            else:
                Rsid = rsid
            LODF = LODF - Rsid

        # reshape results into 1D array if only one line
        if isinstance(line, (int, str)):
            LODF = LODF[:, 0]

        if (not no_store) & (line is None):
            self.LODF._v = LODF
        return LODF

    def build_otdf(self, line=None):
        """
        Build the Outrage Transfer Distribution Factor (OTDF) matrix for line
        k outage: $OTDF_k = PTDF + LODF[:, k] @ PTDF[k, ]$.

        OTDF_k[m, n] means the increased line flow on line `m` when there is
        1 p.u. power injection at bus `n` when line `k` is outage.

        Note that the OTDF is not stored in the MatProcessor.

        Parameters
        ----------
        line : int, str, list, optional
            Lines index for which the OTDF is calculated. It takes both single
            or multiple line indices.
            If not given, the first line is used by default.

        Returns
        -------
        OTDF : np.ndarray, scipy.sparse.csr_matrix
            Line outage distribution factor.

        References
        ----------
        1. PowerWorld Documentation, Line Outage Distribution Factors,
           https://www.powerworld.com/WebHelp/Content/MainDocumentation_HTML/Line_Outage_Distribution_Factors_LODFs.htm
        """
        if (self.PTDF._v is None) or (self.LODF._v is None):
            raise ValueError("Internal PTDF and LODF are not available. Please build them first.")

        ptdf = self.PTDF._v
        lodf = self.LODF._v

        if line is None:
            luid = [0]
        elif isinstance(line, (int, str)):
            try:
                luid = [self.system.Line.idx2uid(line)]
            except ValueError:
                raise ValueError(f"Line {line} not found.")
        elif isinstance(line, list):
            luid = self.system.Line.idx2uid(line)

        otdf = ptdf + lodf[:, luid] @ ptdf[luid, :]
        return otdf
