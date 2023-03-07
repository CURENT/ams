"""
OPF routines.
"""
import logging
from ams.routines.base import BaseRoutine, timer

logger = logging.getLogger(__name__)


class OPF(BaseRoutine):
    """
    AC Optimal Power Flow routine.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.info = "AC Optimal Power Flow"
        self._algeb_models = ['Bus', 'PV', 'Slack']

    @timer
    def _solve(self, **kwargs):
        return None

    def _unpack(self):
        """
        Unpack the results.
        """
        return None

    def run(self, **kwargs):
        """
        Run OPF.
        """
        _, elapsed_time = self._solve(**kwargs)
        self._unpack()
        info = f'{self.class_name} completed in {elapsed_time} seconds with exit code {self.exit_code}.'
        logger.info(info)


class DCOPF(OPF):
    """
    DC Optimal Power flow routine.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.info = "DC Optimal Power Flow"
