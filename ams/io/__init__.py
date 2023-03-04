"""
AMS input parsers and output formatters.
"""

import importlib
import logging

from andes.utils.misc import elapsed
from andes.io import guess

from ams.io import xlsx, psse, matpower, pypower   # NOQA


logger = logging.getLogger(__name__)


def parse(system):
    """
    Parse input file with the given format in `system.files.input_format`.

    Returns
    -------
    bool
        True if successful; False otherwise.
    """

    t, _ = elapsed()

    # exit when no input format is given
    if not system.files.input_format:
        if not guess(system):
            logger.error('Input format unknown for file "%s".', system.files.case)
            return False

    # try parsing the base case file
    logger.info('Parsing input file "%s"...', system.files.case)
    input_format = system.files.input_format
    parser = importlib.import_module('.' + input_format, __name__)
    if not parser.read(system, system.files.case):
        logger.error('Error parsing file "%s" with <%s> parser.', system.files.fullname, input_format)
        return False

    _, s = elapsed(t)
    logger.info('Input file parsed in %s.', s)

    # Try parsing the addfile
    t, _ = elapsed()

    if system.files.addfile:
        logger.info('Parsing additional file "%s"...', system.files.addfile)
        add_format = system.files.add_format
        add_parser = importlib.import_module('.' + add_format, __name__)
        if not add_parser.read_add(system, system.files.addfile):
            logger.error('Error parsing addfile "%s" with %s parser.', system.files.addfile, input_format)
            return False
        _, s = elapsed(t)
        logger.info('Addfile parsed in %s.', s)

    return True
