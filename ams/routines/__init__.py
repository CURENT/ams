"""
Dispatch routines.
"""

from collections import OrderedDict
from andes.utils.func import list_flatten
from ams.routines.routine import Routine
from ams.routines.routinedata import RoutineData

all_routines = OrderedDict([
    ('pflow', ['PFlow', 'DCPF']),
    # ('opf', ['OPF', 'DCOPF']),
    ('dcopf', ['DCOPF']),
])

class_names = list_flatten(list(all_routines.values()))
routine_cli = OrderedDict([(item.lower(), item) for item in class_names])

# TODO: move this definition into each routine ``__init__``
algeb_models = OrderedDict([
    ('PFlow', ['Bus', 'PV', 'Slack']),
    ('DCPF', ['Bus', 'PV', 'Slack']),
    ('OPF', ['Bus', 'PV', 'Slack']),
    ('DCOPF', ['Bus', 'PV', 'Slack']),
])
