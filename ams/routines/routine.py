"""
Module for routine data.
"""

import logging  # NOQA
from collections import OrderedDict  # NOQA

import numpy as np

from andes.core import Config
from ams.core.var import RAlgeb
from andes.shared import deg2rad
from andes.utils.misc import elapsed
from ams.utils import timer
from ams.core.param import RParam
from ams.opt.omodel import OModel, Var, Constraint

from ams.core.symprocessor import SymProcessor

logger = logging.getLogger(__name__)

class Routine:
    """
    CLass to hold routine parameters and variables.
    """

    def __init__(self, system=None, config=None):
        self.system = system
        self.config = Config(self.class_name)

        self.tex_names = OrderedDict((('sys_f', 'f_{sys}'),
                                      ('sys_mva', 'S_{b,sys}'),
                                      ))
        self.syms = SymProcessor(self)  # symbolic processor

        self.ralgebs = OrderedDict()  # list out RAlgebs in a routine
        self.constrs = OrderedDict()
        self.obj = None

        # --- optimization modeling ---
        self.om = OModel(routine=self)

        if config is not None:
            self.config.load(config)
        # TODO: these default configs might to be revised
        self.config.add(OrderedDict((('sparselib', 'klu'),
                                     ('linsolve', 0),
                                     )))
        self.config.add_extra("_help",
                              sparselib="linear sparse solver name",
                              linsolve="solve symbolic factorization each step (enable when KLU segfaults)",
                              )
        self.config.add_extra("_alt",
                              sparselib=("klu", "umfpack", "spsolve", "cupy"),
                              linsolve=(0, 1),
                              )

        self.exec_time = 0.0  # recorded time to execute the routine in seconds
        # TODO: check exit_code of gurobipy or any other similiar solvers
        self.exit_code = 0  # exit code of the routine; 1 for successs

    @property
    def class_name(self):
        return self.__class__.__name__

    def doc(self, max_width=78, export='plain'):
        """
        Routine documentation interface.
        """
        return self.config.doc(max_width, export)

    def setup(self):
        """
        Setup optimization model.
        """
        results, elapsed_time = self.om.setup()
        common_info = f"{self.class_name} model set up "
        if results:
            info = f"in {elapsed_time}."
        else:
            info = "failed!"
        logger.info(common_info + info)
        return results

    def prepare(self):
        """
        Prepare the routine.
        """
        logger.debug("Generating code for %s", self.class_name)
        self.syms.generate_symbols()

    @timer
    def solve(self, **kwargs):
        """
        Solve the routine.
        """
        return None

    def unpack(self, **kwargs):
        """
        Unpack the results.
        """
        return None

    def run(self, **kwargs):
        """
        Routine the routine.
        """
        _, elapsed_time = self.solve(**kwargs)
        self.unpack(**kwargs)
        info = f"{self.class_name} completed in {elapsed_time} with exit code {self.exit_code}."
        logger.info(info)
        return self.exit_code

    def summary(self, **kwargs):
        """
        Summary interface
        """
        raise NotImplementedError

    def report(self, **kwargs):
        """
        Report interface.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"Routine {self.__class__.__name__} at {hex(id(self))}"

    def _ppc2ams(self):
        """
        Convert PYPOWER results to AMS.
        """
        raise NotImplementedError

    def __setattr__(self, key, value):
        """
        Overload the setattr function to register attributes.

        Parameters
        ----------
        key : str
            name of the attribute
        value : [Algeb]
            value of the attribute
        """

        # NOTE: value.id is not in use yet
        if isinstance(value, RAlgeb):
            value.id = len(self.ralgebs)
        elif isinstance(value, RParam):
            value.id = len(self.rparams)
        self._register_attribute(key, value)

        super(Routine, self).__setattr__(key, value)

    def _register_attribute(self, key, value):
        """
        Register a pair of attributes to the model instance.

        Called within ``__setattr__``, this is where the magic happens.
        Subclass attributes are automatically registered based on the variable type.
        Block attributes will be exported and registered recursively.
        """
        if isinstance(value, RAlgeb):
            self.ralgebs[key] = value
        elif isinstance(value, RParam):
            self.rparams[key] = value
        elif isinstance(value, Constraint):
            self.constrs[key] = value