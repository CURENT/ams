"""
Module for system matrix make.
"""

import logging
from typing import Optional

import numpy as np

from scipy.sparse import csr_matrix as c_sparse
from scipy.sparse import lil_matrix as l_sparse

from ams.pypower.make import makePTDF, makeBdc
from ams.io.pypower import system2ppc

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
        Param.__init__(self, name=name, info=info)
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
        if isinstance(self._v, (c_sparse, l_sparse)):
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
        self.PTDF = MParam(name='PTDF', tex_name=r'P_{TDF}',
                           info='Power transfer distribution factor',
                           v=None)
        self.pl = MParam(name='pl', tex_name=r'p_l',
                         info='Nodal active load',
                         v=None, sparse=True)
        self.ql = MParam(name='ql', tex_name=r'q_l',
                         info='Nodal reactive load',
                         v=None, sparse=True)
        self.Cft = MParam(name='Cft', tex_name=r'C_{ft}',
                          info='Connectivity matrix',
                          v=None, sparse=True)
        self.Cfti = MParam(name='Cfti', tex_name=r'C_{ft}^i',
                           info='Connectivity matrix',
                           v=None, sparse=False)
        self.Cg = MParam(name='Cg', tex_name=r'C_g',
                         info='Generator connectivity matrix',
                         v=None, sparse=True)
        self.Cgi = MParam(name='Cgi', tex_name=r'C_g^i',
                          info='Generator connectivity matrix',
                          v=None, sparse=True)
        self.Cs = MParam(name='Cs', tex_name=r'C_s',
                         info='Slack connectivity matrix',
                         v=None, sparse=True)
        self.Csi = MParam(name='Csi', tex_name=r'C_s^i',
                          info='Slack connectivity matrix',
                          v=None, sparse=True)
        self.Cl = MParam(name='Cl', tex_name=r'Cl',
                         info='Load connectivity matrix',
                         v=None, sparse=True)
        self.Cli = MParam(name='Cli', tex_name=r'Cl^i',
                          info='Load connectivity matrix',
                          v=None, sparse=True)

    def make(self):
        """
        Make the system matrices.

        Note that this method will update all matrices in the class.
        """
        system = self.system
        ppc = system2ppc(system)

        self.PTDF._v = makePTDF(ppc['baseMVA'], ppc['bus'], ppc['branch'])
        _, _, _, _, self.Cft._v = makeBdc(ppc['baseMVA'], ppc['bus'], ppc['branch'])
        self.Cfti._v = np.linalg.pinv(self.Cft.v)

        gen_bus = system.StaticGen.get(src='bus', attr='v',
                                       idx=system.StaticGen.get_idx())
        slack_bus = system.Slack.get(src='bus', attr='v',
                                     idx=system.Slack.idx.v)
        all_bus = system.Bus.idx.v
        load_bus = system.StaticLoad.get(src='bus', attr='v',
                                         idx=system.StaticLoad.get_idx())
        idx_PD = system.PQ.find_idx(keys="bus", values=all_bus,
                                    allow_none=True, default=None)
        self.pl._v = c_sparse(system.PQ.get(src='p0', attr='v', idx=idx_PD))
        self.ql._v = c_sparse(system.PQ.get(src='q0', attr='v', idx=idx_PD))

        row, col = np.meshgrid(all_bus, slack_bus)
        Cs_v = (row == col).astype(int)
        self.Cs._v = c_sparse(Cs_v)
        self.Csi._v = c_sparse(np.linalg.pinv(Cs_v))
        row, col = np.meshgrid(all_bus, gen_bus)
        Cg_v = (row == col).astype(int)
        self.Cg._v = c_sparse(Cg_v)
        self.Cgi._v = c_sparse(np.linalg.pinv(Cg_v))
        row, col = np.meshgrid(all_bus, load_bus)
        Cl_v = (row == col).astype(int)
        self.Cl._v = c_sparse(Cl_v)
        self.Cli._v = c_sparse(np.linalg.pinv(Cl_v))

        return True

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
