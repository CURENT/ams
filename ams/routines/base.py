"""
Module for base routine.
"""

import logging  # NOQA
from collections import OrderedDict  # NOQA

import numpy as np

from andes.core import Config
from andes.shared import deg2rad

logger = logging.getLogger(__name__)


class BaseResults:
    """
    Base class for holding dispatch results.
    """

    def __init__(self):
        pass

    def as_df(self):
        """
        Convert the data to a pandas DataFrame.
        """
        pass


class BaseRoutine:
    """
    Base routine class.

    Parameters
    ----------
    system : ams.system
        The AMS system.
    config : dict
        Configuration dict.

    Attributes
    ----------
    system : ams.system
        The AMS system.
    config : andes.core.Config
        Configuration object.
    name : str
        Routine information.
    models : OrderedDict
        Dict that stores all involved devices.
    count : OrderedDict
        Dict that stores the count of all involved devices.
    algebs : OrderedDict
        OrderedDict of list that stores ``name`` and ``idx`` of Algebs from all involved devices.
    v : numpy.ndarray
        Array that stores the values of algebraic variables.
    """

    def __init__(self, system=None, config=None):
        self.system = system
        self.config = Config(self.class_name)
        self.info = None
        # NOTE: the following attributes are populated in ``System`` class
        self.models = OrderedDict()  # collect all involved devices
        self.ralgebs = OrderedDict()  # all routine algebraic variables
        self.n = 0  # number of algebraic variables
        self.v = np.empty(0)  # values of algebraic variables

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

    def _count(self):
        """
        Initialize algebraic variables and set address.
        This method is called in ``System`` after all routiens are imported.

        Parameters
        ----------
        ndevice : int
            number of devices

        Returns
        -------
        n_algeb: int
            number of devices
        mdl_all: list of ams.core.model.Model
            list of all involved devices
        """
        out = OrderedDict()
        n_algeb = 0
        mdl_all = []
        for mname in self.models:
            mdl = getattr(self.system, f'{mname}')  # instance of model
            for var_name in mdl.algebs:
                n_algeb += mdl.n  # number of algebs
                mdl_all.append(mdl.class_name())
        return n_algeb, mdl_all

    def run(self, **kwargs):
        """
        Routine main entry point.
        """
        raise NotImplementedError

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
