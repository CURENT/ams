"""
AMS input parsers and output formatters.
"""

import importlib
import logging

import os

from ams.utils.misc import elapsed

from ams.io import xlsx, psse, matpower, pypower, json   # NOQA

__all__ = ['guess', 'parse', 'dump', 'input_formats', 'output_formats']

logger = logging.getLogger(__name__)

# Input formats is a dictionary of supported format names and the accepted file extensions
# The first file will be parsed by read() function and the addfile will be parsed by read_add()
# Typically, column based formats, such as IEEE CDF and PSS/E RAW, are faster to parse

input_formats = {
    'xlsx': ('xlsx',),
    'json': ('json',),
    'matpower': ('m', ),
    'psse': ('raw', 'dyr'),
    'pypower': ('py',),
    'powerworld': ('aux',),
}

# Output formats is a dictionary of supported output formats and their extensions
# The static data will be written by write() function and the addfile by writeadd()

output_formats = {
    'xlsx': ('xlsx',),
    'json': ('json',),
}


def guess(system):
    """
    Guess the input format based on extension and content.

    Also stores the format name to `system.files.input_format`.

    Parameters
    ----------
    system : System
        System instance with the file name set to `system.files`

    Returns
    -------
    str
        format name
    """
    files = system.files
    maybe = []
    if files.input_format:
        maybe.append(files.input_format)
    # first, guess by extension
    for key, val in input_formats.items():
        if files.ext.strip('.').lower() in val:
            maybe.append(key)

    # second, guess by lines
    true_format = ''

    for item in maybe:
        parser = importlib.import_module('.' + item, __name__)
        testlines = getattr(parser, 'testlines')
        if testlines(files.case):
            true_format = item
            files.input_format = true_format
            logger.debug('Input format guessed as %s.', true_format)
            break

    if not true_format:
        logger.error('Unable to determine case format.')

    # guess addfile format
    if files.addfile:
        _, add_ext = os.path.splitext(files.addfile)
        for key, val in input_formats.items():
            if add_ext[1:] in val:
                files.add_format = key
                logger.debug('Addfile format guessed as %s.', key)
                break

    return true_format


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


def dump(system, output_format, full_path=None, overwrite=None, **kwargs):
    """
    Dump the AMS system data into the requested output format.

    Parameters
    ----------
    system
        System object.
    output_format : str
        Output format name (``'xlsx'`` or ``'json'``). As a convenience,
        ``None`` or ``True`` is treated as ``'xlsx'``.
    full_path : str, optional
        Full path for the output file. Defaults to
        ``<output_path>/<case_name>.<ext>``.
    overwrite : bool or None, optional
        ``None`` (default) prompts interactively when the target exists;
        ``True`` overwrites without prompting; ``False`` refuses to
        overwrite. Matches ``ams.io.xlsx.write`` / ``ams.io.json.write``.
    **kwargs
        Forwarded to the per-format writer. ``add_book`` is xlsx-only and
        is dropped for json.

    Returns
    -------
    bool
        True if successful; False otherwise.
    """
    if output_format is None or output_format is True:
        output_format = 'xlsx'

    if output_format not in output_formats:
        logger.error("Dump format <%s> not supported.", output_format)
        return False

    ext = output_formats[output_format][0]
    if full_path is not None:
        system.files.dump = full_path
    else:
        system.files.dump = os.path.join(system.files.output_path,
                                         system.files.name + '.' + ext)

    t, _ = elapsed()
    ret = False
    if output_format == 'xlsx':
        ret = xlsx.write(system, system.files.dump, overwrite=overwrite, **kwargs)
    elif output_format == 'json':
        # ``add_book`` is xlsx-only; ams.io.json.write has no **kwargs, so strip it.
        json_kwargs = {k: v for k, v in kwargs.items() if k != 'add_book'}
        ret = json.write(system, system.files.dump, overwrite=overwrite, **json_kwargs)
    _, s = elapsed(t)

    if ret:
        logger.info('Format conversion completed in %s.', s)
        return True
    logger.error('Format conversion failed.')
    return False
