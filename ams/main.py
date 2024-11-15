"""
Main entry point for the AMS CLI and scripting interfaces.
"""

import cProfile
import io
import logging
import os
import platform
import pstats
import sys
from functools import partial
from subprocess import call
from time import sleep
from typing import Optional, Union
import textwrap

from ._version import get_versions

from andes.main import _find_cases
from andes.shared import Pool, Process, coloredlogs, unittest, NCPUS_PHYSICAL
from andes.utils.misc import elapsed, is_interactive

import ams
from ams.routines import routine_cli
from ams.system import System
from ams.utils.paths import get_config_path, get_log_dir, tests_root
from ams.shared import copyright_msg

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

    Notes
    -----
    Copied from the ANDES project (https://github.com/CURENT/andes).
    Original author: Hantao Cui
    License: GPL3
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

    Notes
    -----
    Revised from the ANDES project (https://github.com/CURENT/andes).
    Original author: Hantao Cui
    License: GPL3
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


def run_case(case, *, routine='pflow', profile=False,
             convert='', convert_all='', add_book=None,
             **kwargs):
    """
    Run single simulation case for the given full path.
    Use ``run`` instead of ``run_case`` whenever possible.

    Argument ``input_path`` will not be prepended to ``case``.

    Arguments recognizable by ``load`` can be passed to ``run_case``.

    Parameters
    ----------
    case : str
        Full path to the test case
    routine : str, ('pflow', 'tds', 'eig')
        Computation routine to run
    profile : bool, optional
        True to enable profiler
    convert : str, optional
        Format name for case file conversion.
    convert_all : str, optional
        Format name for case file conversion, output
        sheets for all available devices.
    add_book : str, optional
        Name of the device to be added to an excel case
        as a new sheet.

    Notes
    -----
    Revised from the ANDES project (https://github.com/CURENT/andes).
    Original author: Hantao Cui
    License: GPL3
    """

    pr = cProfile.Profile()
    # enable profiler if requested
    if profile is True:
        pr.enable()

    system = load(case,
                  use_input_path=False,
                  **kwargs)

    if system is None:
        return None

    skip_empty = True
    overwrite = None
    # convert to xlsx and process `add-book` option
    if add_book is not None:
        convert = 'xlsx'
        overwrite = True
    if convert_all != '':
        convert = 'xlsx'
        skip_empty = False

    # convert to the requested format
    if convert != '':
        ams.io.dump(system, convert, overwrite=overwrite, skip_empty=skip_empty,
                    add_book=add_book)
        return system

    # run the requested routine
    if routine is not None:
        if isinstance(routine, str):
            routine = [routine]

        if system.is_setup:
            for r in routine:
                system.__dict__[routine_cli[r.lower()]].run(**kwargs)
        else:
            logger.error("System is not set up. Routines cannot continue.")

    # Disable profiler and output results
    if profile:
        pr.disable()

        if system.files.no_output:
            nlines = 40
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=sys.stdout).sort_stats('cumtime')
            ps.print_stats(nlines)
            logger.info(s.getvalue())
            s.close()
        else:
            nlines = 999
            with open(system.files.prof, 'w') as s:
                ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
                ps.print_stats(nlines)
                ps.dump_stats(system.files.prof_raw)
            logger.info('cProfile text data written to "%s".', system.files.prof)
            logger.info('cProfile raw data written to "%s". View with tool `snakeviz`.', system.files.prof_raw)

    return system


def _run_mp_proc(cases, ncpu=NCPUS_PHYSICAL, **kwargs):
    """
    Run multiprocessing with `Process`.

    Return values from `run_case` are not preserved. Always return `True` when done.

    Notes
    -----
    Copied from the ANDES project (https://github.com/CURENT/andes).
    Original author: Hantao Cui
    License: GPL3
    """

    # start processes
    jobs = []
    for idx, file in enumerate(cases):
        job = Process(name=f'Process {idx:d}', target=run_case, args=(file,), kwargs=kwargs)
        jobs.append(job)
        job.start()
        start_msg = f'Process {idx:d} for "{file:s}" started.'
        print(start_msg)
        logger.debug(start_msg)
        if (idx % ncpu == ncpu - 1) or (idx == len(cases) - 1):
            sleep(0.1)
            for job in jobs:
                job.join()
            jobs = []

    return True


def _run_mp_pool(cases, ncpu=NCPUS_PHYSICAL, verbose=logging.INFO, **kwargs):
    """
    Run multiprocessing jobs using Pool.

    This function returns all System instances in a list, but requires longer computation time.

    Parameters
    ----------
    ncpu : int, optional = os.cpu_cout()
        Number of cpu cores to use in parallel
    mp_verbose : 10 - 50
        Verbosity level during multiprocessing
    verbose : 10, 20, 30, 40, 50
        Verbosity level outside multiprocessing

    Notes
    -----
    Copied from the ANDES project (https://github.com/CURENT/andes).
    Original author: Hantao Cui
    License: GPL3
    """

    pool = Pool(ncpu)
    print("Cases are processed in the following order:")
    print('\n'.join([f'"{name}"' for name in cases]))

    ret = pool.map(partial(run_case,
                           verbose=verbose,
                           remove_pycapsule=True,
                           autogen_stale=False,
                           **kwargs),
                   cases)

    # FIXME: does following code work in AMS?
    # # fix address for in-place arrays
    # for ss in ret:
    #     fix_view_arrays(ss)

    return ret


def run(filename, input_path='', verbose=20, mp_verbose=30,
        ncpu=NCPUS_PHYSICAL, pool=False,
        cli=False, shell=False, **kwargs):
    """
    Entry point to run AMS routines.

    Parameters
    ----------
    filename : str
        file name (or pattern)
    input_path : str, optional
        input search path
    verbose : int, 10 (DEBUG), 20 (INFO), 30 (WARNING), 40 (ERROR), 50 (CRITICAL)
        Verbosity level. If ``config_logger`` is called prior to ``run``,
        this option will be ignored.
    mp_verbose : int
        Verbosity level for multiprocessing tasks
    ncpu : int, optional
        Number of cpu cores to use in parallel
    pool: bool, optional
        Use Pool for multiprocessing to return a list of created Systems.
    kwargs
        Other supported keyword arguments
    cli : bool, optional
        If is running from command-line. If True, returns exit code instead of System
    shell : bool, optional
        If True, enter IPython shell after routine.

    Returns
    -------
    System or exit_code
        An instance of system (if `cli == False`) or an exit code otherwise..

    Notes
    -----
    Copied from the ANDES project (https://github.com/CURENT/andes).
    Original author: Hantao Cui
    License: GPL3
    """

    if is_interactive() and len(logger.handlers) == 0:
        config_logger(verbose, file=False)

    # put some args back to `kwargs`
    kwargs['input_path'] = input_path
    kwargs['verbose'] = verbose

    cases = _find_cases(filename, input_path)

    system = None
    ex_code = 0

    if len(filename) > 0 and len(cases) == 0:
        ex_code = 1  # file specified but not found

    t0, _ = elapsed()
    if len(cases) == 1:
        system = run_case(cases[0], **kwargs)
    elif len(cases) > 1:
        # FIXME: after standardize code generation, enable following code
        # # import `pycode` to local namespace to avoid a picking issue
        # import_pycode()

        # suppress logging output during multiprocessing
        logger.info('-> Processing %s jobs on %s CPUs.', len(cases), ncpu)
        set_logger_level(logger, logging.StreamHandler, mp_verbose)
        set_logger_level(logger, logging.FileHandler, logging.DEBUG)

        if pool is True:
            system = _run_mp_pool(cases,
                                  ncpu=ncpu,
                                  mp_verbose=mp_verbose,
                                  **kwargs)
        else:
            system = _run_mp_proc(cases,
                                  ncpu=ncpu,
                                  mp_verbose=mp_verbose,
                                  **kwargs)

        # restore command line output when all jobs are done
        set_logger_level(logger, logging.StreamHandler, verbose)

        log_files = find_log_path(logger)
        if len(log_files) > 0:
            log_paths = '\n'.join(log_files)
            print(f'Log saved to "{log_paths}".')

    t0, s0 = elapsed(t0)

    if len(cases) == 1:
        if system is not None:
            ex_code += system.exit_code
        else:
            ex_code += 1
    elif len(cases) > 1:
        if isinstance(system, list):
            for s in system:
                ex_code += s.exit_code

    if len(cases) == 1:
        if ex_code == 0:
            print(f'-> Single process finished in {s0}.')
        else:
            print(f'-> Single process exit with an error in {s0}.')
    elif len(cases) > 1:
        if ex_code == 0:
            print(f'-> Multiprocessing finished in {s0}.')
        else:
            print(f'-> Multiprocessing exit with an error in {s0}.')

    # IPython interactive shell
    if shell is True:
        try:
            from IPython import embed

            # load plotter before entering IPython
            if system is None:
                logger.warning("IPython: The System object has not been created.")
            elif isinstance(system, System):
                logger.info("IPython: Access System object in variable `system`.")
                system.TDS.load_plotter()
            elif isinstance(system, list):
                logger.warning("IPython: System objects stored in list `system`.\n"
                               "Call `TDS.load_plotter()` on each for plotter.")

            embed()
        except ImportError:
            logger.warning("IPython import error. Installed?")

    if cli is True:
        return ex_code

    return system


def set_logger_level(lg, type_to_set, level):
    """
    Set logging level for the given type of handler.
    """

    for h in lg.handlers:
        if isinstance(h, type_to_set):
            h.setLevel(level)


def find_log_path(lg):
    """
    Find the file paths of the FileHandlers.
    """
    out = []
    for h in lg.handlers:
        if isinstance(h, logging.FileHandler):
            out.append(h.baseFilename)
    return out


def misc(edit_config='', save_config='', show_license=False, clean=True, recursive=False,
         overwrite=None, version=False, **kwargs):
    """
    Miscellaneous commands.

    Notes
    -----
    Copied from the ANDES project (https://github.com/CURENT/andes).
    Original author: Hantao Cui
    License: GPL3
    """

    if edit_conf(edit_config):
        return
    if show_license:
        print_license()
        return
    if save_config != '':
        save_conf(save_config, overwrite=overwrite, **kwargs)
        return
    if clean is True:
        remove_output(recursive)
        return

    if demo is True:
        demo(**kwargs)
        return

    if version is True:
        versioninfo()
        return

    logger.info("info: no option specified. Use 'ams misc -h' for help.")


def doc(attribute=None, list_supported=False, config=False, **kwargs):
    """
    Quick documentation from command-line.

    Notes
    -----
    Revised from the ANDES project (https://github.com/CURENT/andes).
    Original author: Hantao Cui
    License: GPL3
    """
    system = System()
    if attribute is not None:
        if attribute in system.__dict__ and hasattr(system.__dict__[attribute], 'doc'):
            logger.info(system.__dict__[attribute].doc())
        else:
            logger.error('Model <%s> does not exist.', attribute)

    elif list_supported is True:
        logger.info(system.supported_routines())

    else:
        logger.info('info: no option specified. Use \'ams doc -h\' for help.')


def demo(**kwargs):
    """
    TODO: show some demonstrations from CLI.
    """
    raise NotImplementedError("Demos have not been implemented")


def versioninfo():
    """
    Print version info for AMS and dependencies.

    Notes
    -----
    Revised from the ANDES project (https://github.com/CURENT/andes).
    Original author: Hantao Cui
    License: GPL3
    """

    import numpy as np
    import cvxpy
    import andes
    from ams.shared import installed_solvers

    versions = {'Python': platform.python_version(),
                'ams': get_versions()['version'],
                'andes': andes.__version__,
                'numpy': np.__version__,
                'cvxpy': cvxpy.__version__,
                'solvers': ', '.join(installed_solvers),
                }
    maxwidth = max([len(k) for k in versions.keys()])

    for key, val in versions.items():
        if key == 'solvers':
            val = textwrap.fill(val, width=30)  # Wrap solvers to a new line if too long
            val = val.replace('\n', '\n' + ' ' * (maxwidth + 2))
        print(f"{key: <{maxwidth}}  {val}")


def print_license():
    """
    Print out AMS license to stdout.
    """

    print(f"""
    AMS version {ams.__version__}

    {copyright_msg}

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    A copy of the GNU General Public License is included below.
    For further information, see <http://www.gnu.org/licenses/>.
    """)
    return True


def edit_conf(edit_config: Optional[Union[str, bool]] = ''):
    """
    Edit the AMS config file which occurs first in the search path.

    Parameters
    ----------
    edit_config : bool
        If ``True``, try to open up an editor and edit the config file. Otherwise returns.

    Returns
    -------
    bool
        ``True`` is a config file is found and an editor is opened. ``False`` if ``edit_config`` is False.

    Notes
    -----
    Copied from the ANDES project (https://github.com/CURENT/andes).
    Original author: Hantao Cui
    License: GPL3
    """
    ret = False

    # no `edit-config` supplied
    if edit_config == '':
        return ret

    conf_path = get_config_path()

    if conf_path is None:
        logger.info('Config file does not exist. Automatically saving.')
        system = System()
        conf_path = system.save_config()

    logger.info('Editing config file "%s"', conf_path)

    editor = ''
    if edit_config is not None:
        # use `edit_config` as default editor
        editor = edit_config
    else:
        # use the following default editors
        if platform.system() == 'Linux':
            editor = os.environ.get('EDITOR', 'vim')
        elif platform.system() == 'Darwin':
            editor = os.environ.get('EDITOR', 'vim')
        elif platform.system() == 'Windows':
            editor = 'notepad.exe'

    editor_cmd = editor.split()
    editor_cmd.append(conf_path)
    call(editor_cmd)
    ret = True
    return ret


def save_conf(config_path=None, overwrite=None, **kwargs):
    """
    Save the AMS config to a file at the path specified by ``save_config``.
    The save action will not run if ``save_config = ''``.

    Parameters
    ----------
    config_path : None or str, optional, ('' by default)

        Path to the file to save the config file. If the path is an emtpy
        string, the save action will not run. Save to
        `~/.ams/ams.conf` if ``None``.

    Returns
    -------
    bool
        ``True`` is the save action is run. ``False`` otherwise.

    Notes
    -----
    Copied from the ANDES project (https://github.com/CURENT/andes).
    Original author: Hantao Cui
    License: GPL3
    """
    ret = False

    # no ``--save-config ``
    if config_path == '':
        return ret

    if config_path is not None and os.path.isdir(config_path):
        config_path = os.path.join(config_path, 'ams.rc')

    ps = System(**kwargs)
    ps.save_config(config_path, overwrite=overwrite)
    ret = True

    return ret


# TODO: change to AMS output types
def remove_output(recursive=False):
    """
    Remove the outputs generated by AMS, including power flow reports
    ``_out.txt``, time-domain list ``_out.lst`` and data ``_out.dat``,
    eigenvalue analysis report ``_eig.txt``.

    Parameters
    ----------
    recursive : bool
        Recursively clean all subfolders

    Returns
    -------
    bool
        ``True`` is the function body executes with success. ``False``
        otherwise.

    Notes
    -----
    Copied from the ANDES project (https://github.com/CURENT/andes).
    Original author: Hantao Cui
    License: GPL3
    """
    found = False
    cwd = os.getcwd()

    if recursive:
        dirs = [x[0] for x in os.walk(cwd)]
    else:
        dirs = (cwd,)

    for d in dirs:
        for file in os.listdir(d):
            if file.endswith('_eig.txt') or \
                    file.endswith('_out.txt'):
                found = True
                try:
                    os.remove(os.path.join(d, file))
                    logger.info('"%s" removed.', os.path.join(d, file))
                except IOError:
                    logger.error('Error removing file "%s".',
                                 os.path.join(d, file))
    if not found:
        logger.info('No output file found in the working directory.')

    return True


def selftest(quick=False, extra=False, **kwargs):
    """
    Run unit tests.

    Notes
    -----
    Copied from the ANDES project (https://github.com/CURENT/andes).
    Original author: Hantao Cui
    License: GPL3
    """

    # map verbosity level from logging to unittest
    vmap = {1: 3, 10: 3, 20: 2, 30: 1, 40: 1, 50: 1}
    verbose = vmap[kwargs.get('verbose', 20)]

    # skip if quick
    quick_skips = ('test_1_docs', 'test_codegen_inc')

    # extra test naming convention
    extra_test = 'extra_test'

    try:
        logger.handlers[0].setLevel(logging.WARNING)
        sys.stdout = open(os.devnull, 'w')  # suppress print statements
    except IndexError:  # logger not set up
        pass

    # discover test cases
    test_directory = tests_root()
    suite = unittest.TestLoader().discover(test_directory)

    # remove codegen for quick mode
    for test_group in suite._tests:
        for test_class in test_group._tests:
            tests_keep = list()

            if not hasattr(test_class, '_tests'):
                continue
            for t in test_class._tests:
                # skip the extra tests if `extra` is not True
                if (extra is not True) and (extra_test in t._testMethodName):
                    continue

                # skip the ones for `quick`
                if quick is True and (t._testMethodName in quick_skips):
                    continue

                tests_keep.append(t)

            test_class._tests = tests_keep

    unittest.TextTestRunner(verbosity=verbose).run(suite)
    sys.stdout = sys.__stdout__
