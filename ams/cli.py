"""
AMS command-line interface and argument parsers.
"""
import logging
import argparse
import platform

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
    logger.info("AMS  under development")


def main():
    """
    Entry point of the ANDES command-line interface.
    """

    preamble()
    return None