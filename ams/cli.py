"""
AMS command-line interface and argument parsers.
"""
import logging
import argparse
import importlib
import platform
import sys
from time import strftime

from andes.shared import NCPUS_PHYSICAL

from ams.main import config_logger
from ams.utils.paths import get_log_dir
from ams.routines import routine_cli

logger = logging.getLogger(__name__)

command_aliases = {
    'prepare': ['prep'],
    'selftest': ['st'],
}


def create_parser():
    """
    Create a parser for the command-line interface.

    Returns
    -------
    argparse.ArgumentParser
        Parser with all AMS options
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-v', '--verbose',
        help='Verbosity level in 10-DEBUG, 20-INFO, 30-WARNING, '
             'or 40-ERROR.',
        type=int, default=20, choices=(1, 10, 20, 30, 40))

    sub_parsers = parser.add_subparsers(dest='command', help='[run] run simulation routine; '
                                                             '[plot] plot results; '
                                                             '[doc] quick documentation; '
                                                             '[misc] misc. functions; '
                                                             '[prepare] prepare the numerical code; '
                                                             '[selftest] run self test; '
                                        )

    run = sub_parsers.add_parser('run')
    run.add_argument('filename', help='Case file name. Power flow is calculated by default.', nargs='*')
    run.add_argument('-r', '--routine', nargs='*', default=('pflow', ),
                     action='store', help='Simulation routine(s). Single routine or multiple separated with '
                                          'space. Run PFlow by default.',
                     choices=list(routine_cli.keys()))
    run.add_argument('-p', '--input-path', help='Path to case files', type=str, default='')
    run.add_argument('-a', '--addfile', help='Additional files used by some formats.')
    run.add_argument('-o', '--output-path', help='Output path prefix', type=str, default='')
    run.add_argument('-n', '--no-output', help='Force no output of any kind', action='store_true')
    run.add_argument('--ncpu', help='Number of parallel processes', type=int, default=NCPUS_PHYSICAL)
    run.add_argument('-c', '--convert', help='Convert to format.', type=str, default='', nargs='?')
    run.add_argument('-b', '--add-book', help='Add a template workbook for the specified model.', type=str)
    run.add_argument('--convert-all', help='Convert to format with all templates.', type=str, default='',
                     nargs='?')
    run.add_argument('--profile', action='store_true', help='Enable Python cProfiler')
    run.add_argument('-s', '--shell', action='store_true', help='Start in IPython shell')
    run.add_argument('--no-preamble', action='store_true', help='Hide preamble')
    run.add_argument('--no-pbar', action='store_true', help='Hide progress bar for time-domain')
    run.add_argument('--pool', action='store_true', help='Start multiprocess with Pool '
                                                         'and return a list of Systems')
    run.add_argument('--from-csv', help='Use data from a CSV file instead of from simulation')
    run.add_argument('-O', '--config-option',
                     help='Set configuration option specificied by '
                          'NAME.FIELD=VALUE with no space. For example, "TDS.tf=2"',
                     type=str, default='', nargs='*')

    doc = sub_parsers.add_parser('doc')
    # TODO: fit to AMS
    # doc.add_argument('attribute', help='System attribute name to get documentation', nargs='?')
    # doc.add_argument('--config', '-c', help='Config help')
    doc.add_argument('--list', '-l', help='List supported models and groups', action='store_true',
                     dest='list_supported')

    misc = sub_parsers.add_parser('misc')
    config_exclusive = misc.add_mutually_exclusive_group()
    config_exclusive.add_argument('--edit-config', help='Quick edit of the config file',
                                  default='', nargs='?', type=str)
    config_exclusive.add_argument('--save-config', help='save configuration to file name',
                                  nargs='?', type=str, default='')
    misc.add_argument('--license', action='store_true', help='Display software license', dest='show_license')
    misc.add_argument('-C', '--clean', help='Clean output files', action='store_true')
    misc.add_argument('-r', '--recursive', help='Recursively clean outputs (combined useage with --clean)',
                      action='store_true')
    # TODO: fit to AMS
    misc.add_argument('-O', '--config-option',
                      help='Set configuration option specificied by '
                      'NAME.FIELD=VALUE with no space. For example, "TDS.tf=2"',
                      type=str, default='', nargs='*')
    misc.add_argument('--version', action='store_true', help='Display version information')

    # TODO: add quick or extra options for selftest
    selftest = sub_parsers.add_parser('selftest', aliases=command_aliases['selftest'])  # NOQA

    return parser


def preamble():
    """
    Log the AMS command-line preamble at the `logging.INFO` level
    """
    from ams import __version__ as version

    py_version = platform.python_version()
    system_name = platform.system()
    date_time = strftime('%m/%d/%Y %I:%M:%S %p')
    logger.info("\n"
                rf"    _             | Version {version}" + '\n'
                rf"   /_\  _ __  ___ | Python {py_version} on {system_name}, {date_time}" + '\n'
                r"  / _ \| '  \(_-< | " + "\n"
                r' /_/ \_\_|_|_/__/ | This program comes with ABSOLUTELY NO WARRANTY.' + '\n')


def main():
    """
    Entry point of the ANDES command-line interface.
    """

    parser = create_parser()
    args = parser.parse_args()

    # Set up logging
    config_logger(stream=True,
                  stream_level=args.verbose,
                  file=True,
                  log_path=get_log_dir(),
                  )
    logger.debug(args)

    module = importlib.import_module('ams.main')

    if args.command in ('plot', 'doc', 'misc'):
        pass
    elif args.command == 'run' and args.no_preamble is True:
        pass
    else:
        preamble()

    # Run the command
    if args.command is None:
        parser.parse_args(sys.argv.append('--help'))

    else:
        cmd = args.command
        for fullcmd, aliases in command_aliases.items():
            if cmd in aliases:
                cmd = fullcmd

        func = getattr(module, cmd)
        return func(cli=True, **vars(args))
