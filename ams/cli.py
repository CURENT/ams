"""
AMS command-line interface and argument parsers.
"""
import logging
import argparse
import importlib
import platform
import sys
from time import strftime

from ams.main import config_logger, find_log_path
from ams.utils.paths import get_log_dir

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
                rf"     _             | Version {version}" + '\n'
                rf"    /_\  _ __  ___ | Python {py_version} on {system_name}, {date_time}" + '\n'
                r"   / _ \| '  \(_-< | " + "\n"
                r'  /_/ \_\_|_|_/__/ | This program comes with ABSOLUTELY NO WARRANTY.' + '\n')

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
    return func(cli=True, **vars(args))
