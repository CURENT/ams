"""
Module for report generation.
"""
import logging
from collections import OrderedDict
from datetime import datetime
from time import strftime

from andes.io.txt import dump_data
from andes.shared import np
from andes.utils.misc import elapsed

from ams import __version__ as version
from ams.shared import copyright_msg

logger = logging.getLogger(__name__)


def report_info(system) -> list:
    info = list()
    info.append('AMS' + ' ' + version + '\n')
    info.append(f'{copyright_msg}\n\n')
    info.append('AMS comes with ABSOLUTELY NO WARRANTY\n')
    info.append('Case file: ' + str(system.files.case) + '\n')
    info.append('Report time: ' + strftime("%m/%d/%Y %I:%M:%S %p") + '\n\n')
    return info
