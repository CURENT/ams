"""
Module for non-linear power flow solving.
"""
import logging

from collections import OrderedDict


logger = logging.getLogger(__name__)


class PFlowSolver:
    """
    Base class for power flow solver.
    """

    def __init__(self, routine):
        self.rtn = routine
        self.prob = None
        self.exprs = OrderedDict()
        self.params = OrderedDict()
        self.vars = OrderedDict()
        self.constrs = OrderedDict()
        self.obj = None
        self.parsed = False
        self.evaluated = False
        self.finalized = False

    @property
    def initialized(self):
        """
        Return the initialization status.
        """
        return self.parsed and self.evaluated and self.finalized

    def parse(self, force=False):
        raise NotImplementedError

    def evaluate(self, force=False):
        raise NotImplementedError

    def finalize(self, force=False):
        raise NotImplementedError

    def init(self):
        """
        Initialize the power flow solver.
        """
        raise NotImplementedError

    def update(self, params):
        raise NotImplementedError

    @property
    def class_name(self):
        """
        Return the class name
        """
        return self.__class__.__name__
