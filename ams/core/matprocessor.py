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
        self.Cft = MParam(name='Cft', tex_name=r'C_{ft}',
                          info='Connectivity matrix',
                          v=None, sparse=True)
        self.Cg = MParam(name='Cg', tex_name=r'C_g',
                         info='Generator connectivity matrix',
                         v=None, sparse=True)
        self.Cs = MParam(name='Cs', tex_name=r'C_s',
                         info='Slack connectivity matrix',
                         v=None, sparse=True)
        self.Cl = MParam(name='Cl', tex_name=r'Cl',
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

        gen_bus = system.StaticGen.get(src='bus', attr='v',
                                       idx=system.StaticGen.get_idx())
        slack_bus = system.Slack.get(src='bus', attr='v',
                                     idx=system.Slack.idx.v)
        all_bus = system.Bus.idx.v
        load_bus = system.StaticLoad.get(src='bus', attr='v',
                                         idx=system.StaticLoad.get_idx())

        gen_u = system.StaticGen.get(src='u', attr='v',
                                     idx=system.StaticGen.get_idx())
        gen_uid_off = np.where(gen_u == 0)[0]

        slack_u = system.Slack.get(src='u', attr='v',
                                   idx=system.Slack.idx.v)
        slack_uid_off = np.where(slack_u == 0)[0]

        load_u = system.StaticLoad.get(src='u', attr='v',
                                       idx=system.StaticLoad.get_idx())
        load_uid_off = np.where(load_u == 0)[0]

        row, col = np.meshgrid(all_bus, slack_bus)
        Cs_v = (row == col).astype(int)
        Cs_v[slack_uid_off, :] = 0
        self.Cs._v = c_sparse(Cs_v)

        row, col = np.meshgrid(all_bus, gen_bus)
        Cg_v = (row == col).astype(int)
        Cg_v[gen_uid_off, :] = 0
        self.Cg._v = c_sparse(Cg_v)

        row, col = np.meshgrid(all_bus, load_bus)
        Cl_v = (row == col).astype(int)
        Cl_v[load_uid_off, :] = 0
        self.Cl._v = c_sparse(Cl_v)

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
