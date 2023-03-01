"""
Base module for routines.
"""

import logging  # NOQA
from collections import OrderedDict  # NOQA

import numpy as np

from andes.core import Config

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
    """

    def __init__(self, system=None, config=None):
        self.system = system
        self.config = Config(self.class_name)
        # NOTE: the following attributes are populated in ``System`` class
        self.algebs = OrderedDict()  # collect algebraic variables from all involved devices
        self.models = OrderedDict()  # collect all involved devices
        self.count = OrderedDict()  # count of all involved devices
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
        self.exit_code = 0  # exit code of the routine; 0 for successs
        self.algebs = OrderedDict()  # internal algebraic variables

    @property
    def class_name(self):
        return self.__class__.__name__

    def doc(self, max_width=78, export='plain'):
        """
        Routine documentation interface.
        """
        return self.config.doc(max_width, export)

    def count_algeb(self):
        """
        Initialize algebraic variables and set address.
        This method is called in ``System`` after all routiens are imported.

        Parameters
        ----------
        ndevice : int
            number of devices

        Returns
        -------
        out: OrderedDict
            an OrderedDict of algebraic variables and their length
        """
        out = OrderedDict()
        for mdl_name in self.models:
            mdl = self.models[mdl_name]  # instance of model
            n = mdl.n  # number of devices
            for var_name in mdl.algebs:
                out[f'{var_name}_{mdl_name}'] = n
        return out

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
