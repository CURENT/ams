"""
Main entry point for the AMS CLI and scripting interfaces.
"""

import logging
import os
from andes.main import set_logger_level, _find_cases
from andes.shared import coloredlogs
from andes.utils.misc import elapsed, is_interactive

import ams
from ams.system import System
from ams.utils.paths import get_config_path, get_log_dir, tests_root
from ams.solver.ipp import system2ppc

logger = logging.getLogger(__name__)


def config_logger(stream_level=logging.INFO, *,
                  stream=True,
                  file=True,
                  log_file='ams.log',
                  log_path=None,
                  file_level=logging.DEBUG,
                  ):
    """
    Configure an AMS logger with a `FileHandler` and a `StreamHandler`.

    This function is called at the beginning of ``ams.main.main()``.
    Updating ``stream_level`` and ``file_level`` is now supported.

    Parameters
    ----------
    stream : bool, optional
        Create a `StreamHandler` for `stdout` if ``True``.
        If ``False``, the handler will not be created.
    file : bool, optionsl
        True if logging to ``log_file``.
    log_file : str, optional
        Logg file name for `FileHandler`, ``'ams.log'`` by default.
        If ``None``, the `FileHandler` will not be created.
    log_path : str, optional
        Path to store the log file. By default, the path is generated by
        get_log_dir() in utils.misc.
    stream_level : {10, 20, 30, 40, 50}, optional
        `StreamHandler` verbosity level.
    file_level : {10, 20, 30, 40, 50}, optional
        `FileHandler` verbosity level.
    Returns
    -------
    None
    """
    lg = logging.getLogger('ams')
    lg.setLevel(logging.DEBUG)

    if log_path is None:
        log_path = get_log_dir()

    sh_formatter_str = '%(message)s'
    if stream_level == 1:
        sh_formatter_str = '%(name)s:%(lineno)d - %(levelname)s - %(message)s'
        stream_level = 10

    sh_formatter = logging.Formatter(sh_formatter_str)
    if len(lg.handlers) == 0:

        # create a StreamHandler
        if stream is True:
            sh = logging.StreamHandler()
            sh.setFormatter(sh_formatter)
            sh.setLevel(stream_level)
            lg.addHandler(sh)

        # file handler for level DEBUG and up
        if file is True and (log_file is not None):
            log_full_path = os.path.join(log_path, log_file)
            fh_formatter = logging.Formatter('%(process)d: %(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh = logging.FileHandler(log_full_path)
            fh.setLevel(file_level)
            fh.setFormatter(fh_formatter)
            lg.addHandler(fh)

        globals()['logger'] = lg

    else:
        # update the handlers
        set_logger_level(logger, logging.StreamHandler, stream_level)
        set_logger_level(logger, logging.FileHandler, file_level)

    if not is_interactive():
        coloredlogs.install(logger=lg, level=stream_level, fmt=sh_formatter_str)


# TODO: check ``load`` later on to see if some of them can be removed
def load(case, setup=True,
         use_input_path=True,
         **kwargs):
    """
    Load a case and set up a system without running routine.
    Return a system.

    Takes other kwargs recognizable by ``System``,
    such as ``addfile``, ``input_path``, and ``no_putput``.

    Parameters
    ----------
    case: str
        Path to the test case
    setup : bool, optional
        Call `System.setup` after loading
    use_input_path : bool, optional
        True to use the ``input_path`` argument to behave
        the same as ``ams.main.run``.

    Warnings
    --------
    If one need to add devices in addition to these from the case
    file, do ``setup=False`` and call ``System.add()`` to add devices.
    When done, manually invoke ``setup()`` to set up the system.
    """
    if use_input_path:
        input_path = kwargs.get('input_path', '')
        case = _find_cases(case, input_path)
        if len(case) > 1:
            logger.error("`ams.load` does not support mulitple cases.")
            return None
        elif len(case) == 0:
            logger.error("No valid case found.")
            return None
        case = case[0]

    system = System(case, **kwargs)

    if not ams.io.parse(system):
        return None

    if setup:
        system.setup()
    return system

def find_log_path(lg):
    """
    Find the file paths of the FileHandlers.
    """
    out = []
    for h in lg.handlers:
        if isinstance(h, logging.FileHandler):
            out.append(h.baseFilename)
    return out
