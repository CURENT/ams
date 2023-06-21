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

from ams.models.group import GroupBase
from ams.core.model import Model

logger = logging.getLogger(__name__)


class RoutineData:
    """
    Class to hold routine parameters.
    """

    def __init__(self):
        self.rparams = OrderedDict()  # list out RParam in a routine


class RoutineModel:
    """
    Class to hold descriptive routine models and data mapping.
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
        self.exit_code = 0  # exit code of the routine;

        self.is_ac = False  # whether the routine is smooth

    @property
    def class_name(self):
        return self.__class__.__name__

    def _loc(self, src: str, idx, allow_none=False):
        """
        Helper function to index a variable or parameter in a routine.
        """
        src_idx = self.__dict__[src].get_idx()
        loc = [src_idx.index(idxe) if idxe in src_idx else None for idxe in idx]
        if None not in loc:
            return loc
        else:
            idx_none = [idxe for idxe in idx if idxe not in src_idx]
            raise ValueError(f'Var <{self.class_name}.{src}> does not contain value with idx={idx_none[0]}')

    def get(self, src: str, idx, attr: str = 'v', allow_none=False, default=0.0):
        """
        Get the value of a variable or parameter.
        """
        if self.__dict__[src].owner is not None:
            owner = self.__dict__[src].owner
            if src in self.map2[owner.class_name].keys():
                src_map = self.map2[owner.class_name][src]
                logger.debug(f'Var <{self.class_name}.{src}> is mapped to <{src_map}> of {owner.class_name}.')
                try:
                    out = owner.get(src=src_map, idx=idx, attr=attr,
                                    allow_none=allow_none, default=default)
                    return out
                except ValueError:
                    raise ValueError(f'Failed to get value of <{src_map}> from {owner.class_name}.')
            else:
                logger.warning(f'Var {self.class_name}.{src} has no mapping to a model or group, try search in routine {self.class_name}.')
                loc = self._loc(src=src, idx=idx, allow_none=allow_none)
                src_v = self.__dict__[src].v
                out = [src_v[l] for l in loc]
                return np.array(out)
        else:
            logger.info(f'Var {self.class_name}.{src} has no owner.')
            # FIXME: add idx for non-grouped variables
            return None

    def set(self, src: str, idx, attr: str = 'v', value=0.0):
        """
        Set the value of an attribute of a routine parameter.
        """
        if self.__dict__[src].owner is not None:
            owner = self.__dict__[src].owner
            try:
                owner.set(src=src, idx=idx, attr=attr, value=value)
                return True
            except KeyError:
                # TODO: hold values to _v if necessary in the future
                logger.info(f'Variable {self.name} has no mapping.')
                return None
        else:
            logger.info(f'Variable {self.name} has no owner.')
            # FIXME: add idx for non-grouped variables
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
        no_input = []
        owner_list = []
        for rname, rparam in self.rparams.items():
            if rparam.owner is not None:
                if rparam.owner.n == 0:
                    no_input.append(rname)
                    owner_list.append(rparam.owner.class_name)
        if len(no_input) > 0:
            logger.error(f"Following models are missing from input file: {set(owner_list)}")
            return False
        # TODO: add data validation for RParam, typical range, etc.
        return True

    def setup(self):
        """
        Setup optimization model.
        """
        # TODO: add input check, e.g., if GCost exists
        if self._data_check():
            logger.debug(f'{self.class_name} data check passed.')
        else:
            logger.error(f'{self.class_name} data check failed, setup may run into error!')
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

    def dc2ac(self, **kwargs):
        """
        Convert the DC-based results with ACOPF.
        """
        raise NotImplementedError

    def _check_attribute(self, key, value):
        """
        Check the attribute pair for valid names while instantiating the class.

        This function assigns `owner` to the model itself, assigns the name and tex_name.
        """
        if key in self.__dict__:
            if key in self.constrs.keys() or key in self.vars.keys():
                logger.warning(f"{self.class_name}: redefinition of member <{key}>. Likely a modeling error.")

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
        self._check_attribute(key, value)
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
