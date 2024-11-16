"""
Module for non-linear power flow solving.
"""
import logging

from ams.opt import OModel


logger = logging.getLogger(__name__)


class PFModel(OModel):
    """
    Base class for power flow solver.
    """

    def __init__(self, routine):
        OModel.__init__(self, routine)

    # TODO: temporary override for development
    def parse(self, force=False):
        self.parsed = True
        return self.parsed

    def evaluate(self, force=False):
        self.evaluated = True
        return self.evaluated

    def finalize(self, force=False):
        self.finalized = True
        return self.finalized

    def init(self, force=False):
        """
        Initialize the power flow solver.
        """
        self.parse(force)
        self.evaluate(force)
        self.finalize(force)
        return self.initialized

    def update(self, params):
        pass

    @property
    def class_name(self):
        """
        Return the class name
        """
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f'{self.rtn.class_name}.{self.__class__.__name__} at {hex(id(self))}'
