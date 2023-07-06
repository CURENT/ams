"""
Module for routine data.
"""

import logging  # NOQA
from collections import OrderedDict  # NOQA

import numpy as np  # NOQA

from andes.core import Config  # NOQA
from andes.shared import deg2rad  # NOQA
from andes.utils.misc import elapsed  # NOQA
from ams.utils import timer  # NOQA
from ams.core.param import RParam  # NOQA
from ams.opt.omodel import OModel, Var, Constraint, Objective  # NOQA

from ams.core.symprocessor import SymProcessor  # NOQA
from ams.core.documenter import RDocumenter  # NOQA
from ams.core.service import RBaseService  # NOQA

from ams.models.group import GroupBase  # NOQA
from ams.core.model import Model  # NOQA

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

        self.services = OrderedDict()  # list out services in a routine

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

        self.is_ac = False  # whether the routine is converted to AC

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
                logger.debug(f'Var <{self.class_name}.{src}> is mapped from <{owner.class_name}.{src_map}>.')
                try:
                    out = owner.get(src=src_map, idx=idx, attr=attr,
                                    allow_none=allow_none, default=default)
                    return out
                except ValueError:
                    raise ValueError(f'Failed to get value of <{src_map}> from {owner.class_name}.')
            else:
                logger.warning(
                    f'Var {self.class_name}.{src} has no mapping to a model or group, try search in routine {self.class_name}.')
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
        # --- setup check ---
        if not self.is_setup:
            logger.info(f"Setup model of {self.class_name}")
            self.setup()
        # NOTE: if the model data is altered, we need to re-setup the model
        # this implementation if not efficient at large-scale
        # FIXME: find a more efficient way to update the OModel values if
        # the system data is altered
        elif self.exec_time > 0:
            self.setup()
        # --- solve optimization ---
        t0, _ = elapsed()
        result = self.solve(**kwargs)
        status = self.om.mdl.status
        self.exit_code = self.syms.status[status]
        self.system.exit_code = self.exit_code
        _, s = elapsed(t0)
        self.exec_time = float(s.split(' ')[0])
        self.unpack(**kwargs)
        if self.exit_code == 0:
            info = f"{self.class_name} solved as {status} in {s} with exit code {self.exit_code}."
            logger.warning(info)
            return True
        else:
            info = f"{self.class_name} failed as {status} with exit code {self.exit_code}!"
            logger.warning(info)
            return False

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
            if hasattr(self, 'constrs'):
                if key in self.constrs.keys() or key in self.vars.keys():
                    logger.warning(f"{self.class_name}: redefinition of member <{key}>. Likely a modeling error.")

        # register owner routine instance of following attributes
        if isinstance(value, (RBaseService)):
            value.rtn = self

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
        if isinstance(value, (Var, Constraint, Objective)):
            value.om = self.om
        if isinstance(value, Var):
            self.vars[key] = value
        elif isinstance(value, Constraint):
            self.constrs[key] = value
        elif isinstance(value, RParam):
            self.rparams[key] = value
        elif isinstance(value, RBaseService):
            self.services[key] = value

    def __delattr__(self, name):
        """
        Overload the delattr function to unregister attributes.

        Parameters
        ----------
        name: str
            name of the attribute
        """
        self._unregister_attribute(name)
        if name == 'obj':
            self.obj = None
        else:
            super().__delattr__(name)  # Call the superclass implementation

    def _unregister_attribute(self, name):
        """
        Unregister a pair of attributes from the routine instance.

        Called within ``__delattr__``, this is where the magic happens.
        Subclass attributes are automatically unregistered based on the variable type.
        """
        if name in self.vars:
            del self.vars[name]
            if name in self.om.vars:
                del self.om.vars[name]
        elif name in self.rparams:
            del self.rparams[name]
        elif name in self.constrs:
            del self.constrs[name]
            if name in self.om.constrs:
                del self.om.constrs[name]
        elif name in self.services:
            del self.services[name]

    def disable(self, name):
        """
        Disable a constraint by name.
        """
        if name in self.constrs:
            if not self.constrs[name].is_enabled:
                logger.warning(f"Constraint <{name}> has already been disabled.")
                return True
            self.is_setup = False
            self.constrs[name].is_enabled = False
            logger.warning(f"Constraint <{name}> is disabled.")
            return True
        else:
            logger.warning(f"Constraint <{name}> not found.")
            return False
