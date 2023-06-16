"""
Module for routine data.
"""

import logging  # NOQA
from collections import OrderedDict  # NOQA

import numpy as np

from andes.core import Config
from andes.shared import deg2rad
from andes.utils.misc import elapsed
from ams.utils import timer
from ams.core.param import RParam
from ams.opt.omodel import OModel, Var, Constraint, Objective

from ams.core.symprocessor import SymProcessor
from ams.core.documenter import RDocumenter

logger = logging.getLogger(__name__)


class RoutineData:
    """
    CLass to hold routine parameters and variables for a dispatch model.
    """

    def __init__(self):
        self.rparams = OrderedDict()  # list out RParam in a routine


class RoutineModel:
    """
    CLass to hold routine parameters and variables.
    """

    def __init__(self, system=None, config=None):
        self.system = system
        self.config = Config(self.class_name)
        self.info = None
        self.tex_names = OrderedDict((('sys_f', 'f_{sys}'),
                                      ('sys_mva', 'S_{b,sys}'),
                                      ))
        self.syms = SymProcessor(self)  # symbolic processor

        self.vars = OrderedDict()  # list out Vars in a routine
        self.constrs = OrderedDict()
        self.obj = None
        self.is_setup = False
        self.type = 'UndefinedType'
        self.docum = RDocumenter(self)

        # --- sync mapping ---
        self.map1 = OrderedDict()  # from ANDES
        self.map2 = OrderedDict()  # to ANDES

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

    def get(self, src: str, idx, attr: str = 'v', allow_none=False, default=0.0):
        """
        Get the value of a variable or parameter.
        """
        if self.__dict__[src].owner is not None:
            owner = self.__dict__[src].owner
            try:
                src_map = self.map2[owner.class_name][src]
                return owner.get(src=src_map,
                                idx=idx,
                                attr=attr, allow_none=allow_none, default=default)
            except KeyError:
                logger.info(f'Variable {self.name} has no mapping.')
                return None
        else:
            logger.info(f'Variable {self.name} has no owner.')
            return None

    def doc(self, max_width=78, export='plain'):
        """
        Retrieve routine documentation as a string.
        """
        return self.docum.get(max_width=max_width, export=export)

    def _data_check(self):
        """
        Check if data is valid for a routine.
        """
        pass
        return True

    def setup(self):
        """
        Setup optimization model.
        """
        # TODO: add input check, e.g., if GCost exists
        self._data_check()
        results, elapsed_time = self.om.setup()
        common_info = f"{self.class_name} model set up "
        if results:
            info = f"in {elapsed_time}."
            self.is_setup = True
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

    def solve(self, **kwargs):
        """
        Solve the routine optimization model.
        """
        pass
        return True

    def unpack(self, **kwargs):
        """
        Unpack the results.
        """
        return None

    def run(self, **kwargs):
        """
        Run the routine.
        """
        if not self.is_setup:
            logger.info(f"Setup model of {self.class_name}")
            self.setup()
        t0, _ = elapsed()
        result = self.solve(**kwargs)
        _, s = elapsed(t0)
        self.exec_time = float(s.split(' ')[0])
        self.unpack(**kwargs)
        info = f"{self.class_name} completed in {s} with exit code {self.exit_code}."
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

    def __repr__(self):
        return f'{self.class_name} at {hex(id(self))}'

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
        key: str
            name of the attribute
        value:
            value of the attribute
        """

        # NOTE: value.id is not in use yet
        if isinstance(value, Var):
            value.id = len(self.vars)
        elif isinstance(value, RParam):
            value.id = len(self.rparams)
        self._register_attribute(key, value)

        super(RoutineModel, self).__setattr__(key, value)

    def _register_attribute(self, key, value):
        """
        Register a pair of attributes to the routine instance.

        Called within ``__setattr__``, this is where the magic happens.
        Subclass attributes are automatically registered based on the variable type.
        """
        if isinstance(value, Var):
            self.vars[key] = value
        elif isinstance(value, RParam):
            self.rparams[key] = value
        elif isinstance(value, Constraint):
            self.constrs[key] = value
