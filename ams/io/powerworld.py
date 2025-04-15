"""
PowerWorld .aux file parser.
"""

import logging
import re

import shlex

import numpy as np  # NOQA
import pandas as pd

logger = logging.getLogger(__name__)


def testlines(infile):
    """
    Test if this file is in the PowerWorld .aux format.
    
    NOT YET IMPLEMENTED.
    """

    return True


def read(system, file):
    """
    Read a PowerWorld .aux data file into a PowerWorld dictionary (ppd),
    and build AMS device elements.
    """

    ppd = aux2ppd(file)
    return ppd2system(ppd, system)


def aux2ppd(infile: str) -> dict:
    """
    Parse a PowerWorld .aux file and return its contents as a structured dictionary.

    This function reads a PowerWorld .aux file and organizes its data into a dictionary
    (`ppd`), where each section of the file is represented as a Pandas DataFrame. The
    column names for each section are extracted from the file's header, and the data
    rows are parsed accordingly.

    The function handles:
    - Multi-line headers for sections
    - Sections with varying numbers of columns
    - Missing or extra data in rows (fills missing values with `None` and truncates extra values)
    - Skips comments and empty lines

    Parameters
    ----------
    infile : str
        Path to the PowerWorld .aux file to be parsed.

    Returns
    -------
    dict
        A dictionary where keys are section names (e.g., "Substation", "Bus") and values
        are Pandas DataFrames containing the data for each section. Each DataFrame's
        columns correspond to the fields defined in the section header.

    Notes
    -----
    - If a section contains more columns than expected, the extra columns are truncated.
    - If a section contains fewer columns than expected, missing values are filled with `None`.
    - The function logs warnings for mismatched column counts and errors during parsing.

    Example
    -------
    >>> ppd = aux2ppd("Hawaii40.AUX")
    >>> print(ppd.keys())
    dict_keys(['Substation', 'Bus', 'Gen', ...])
    >>> print(ppd['Substation'].head())
       Number     Name IDExtra   Latitude   Longitude DataMaintainerAssign DataMaintainerInheritBlock
    0       1    ALOHA       1  21.344246 -157.939889                                        
    1       2   FLOWER       2  21.347927 -157.876274                                        
    2       3     WAVE       3  21.293143 -157.848767                                        
    3       4  HONOLULU       4  21.316548 -157.845053                                        
    4       5     SURF       5  21.291518 -157.826869                                        
    """
    
    ppd = {}
    headers = {}

    current_section = None
    current_data = []

    accumulating_header = False
    header_lines = []

    with open(infile, 'r') as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith('//'):
                continue

            if '(' in line and not line.endswith('{'):
                accumulating_header = True
                header_lines = [line]
                continue

            if accumulating_header:
                header_lines.append(line)
                if line.endswith('{'):
                    full_header = ' '.join(header_lines)
                    match = re.match(r'^([\w\s]+)\s*\(([^)]*)\)\s*{', full_header)
                    if match:
                        if current_section is not None:
                            df = pd.DataFrame(current_data, columns=headers[current_section])
                            ppd[current_section] = df
                        current_section = match.group(1).strip()
                        col_string = match.group(2).strip()
                        headers[current_section] = [col.strip() for col in col_string.split(',')]
                        current_data = []
                    accumulating_header = False
                continue

            if line == '}':
                if current_section is not None:
                    df = pd.DataFrame(current_data, columns=headers[current_section])
                    ppd[current_section] = df
                    current_section = None
                    current_data = []
                continue

            if current_section:
                try:
                    tokens = shlex.split(line)
                    expected_cols = len(headers[current_section])
                    actual_cols = len(tokens)

                    if actual_cols < expected_cols:
                        tokens += [None] * (expected_cols - actual_cols)
                    elif actual_cols > expected_cols:
                        logger.debug(
                            f"[Line {lineno}] Too many columns in section '{current_section}': "
                            f"expected {expected_cols}, got {actual_cols}. Truncating."
                        )
                        tokens = tokens[:expected_cols]

                    current_data.append(tokens)

                except Exception as e:
                    logger.error(f"[Line {lineno}] Parse error: {e}\n  Line: {line}")

    if current_section is not None:
        df = pd.DataFrame(current_data, columns=headers[current_section])
        ppd[current_section] = df

    return ppd


def ppd2system(ppd: dict, system) -> bool:
    """
    Load a PowerWorld dictionary (ppd) into the AMS system object.

    Parameters
    ----------
    ppd : dict
        PowerWorld dictionary (ppd).
    system : ams.System.system
        AMS system object.

    Returns
    -------
    bool
        True if successful; False otherwise.
    """
    # TODO: implement

    return True
