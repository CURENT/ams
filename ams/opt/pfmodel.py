"""
Module for non-linear power flow solving.
"""
import logging

from ams.opt.omodel import OModelBase


logger = logging.getLogger(__name__)


class PFModel(OModelBase):
    """
    Base class for power flow solver.
    """

    def __init__(self, routine):
        OModelBase.__init__(self, routine)
