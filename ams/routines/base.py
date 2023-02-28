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
        self.algebs = OrderedDict()  # collect algebraic variables from all involved devices

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

    def request_address(self, ndevice, nvar, collate=False):
        """
        Interface for requesting addresses for a model.

        Parameters
        ----------
        ndevice : int
            number of devices
        nvar : int
            number of variables
        collate : bool, optional
            False if the same variable for different devices are contiguous.
            True if variables for the same devices should collate. Note: setting
            ``collate`` to True will degrade the performance.

        Returns
        -------
        list
            A list of arrays for each variable.
        """

        out = []
        counter_name = 'm'

        idx_begin = self.__dict__[counter_name]
        idx_end = idx_begin + ndevice * nvar

        if not collate:
            for idx in range(nvar):
                out.append(np.arange(idx_begin + idx * ndevice, idx_begin + (idx + 1) * ndevice))
        else:
            for idx in range(nvar):
                out.append(np.arange(idx_begin + idx, idx_end, nvar))

        self.__dict__[counter_name] = idx_end

        return out

    @property
    def class_name(self):
        return self.__class__.__name__

    def doc(self, max_width=78, export='plain'):
        """
        Routine documentation interface.
        """
        return self.config.doc(max_width, export)

    def init(self):
        """
        Routine initialization interface.
        """
        pass

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


class pf(BaseRoutine):
    """
    Power flow routine.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)

    def run(self, **kwargs):
        """
        Run power flow.
        """
        pass

    def summary(self, **kwargs):
        """
        Print power flow summary.
        """
        pass

    def report(self, **kwargs):
        """
        Print power flow report.
        """
        pass
