"""
Dispatch routines.
"""

from collections import OrderedDict
from andes.utils.func import list_flatten
from ams.routines.routine import RoutineData, RoutineModel

all_routines = OrderedDict([
    ('dcpf', ['DCPF']),
    ('pflow', ['PFlow']),
    ('acopf', ['ACOPF']),
    ('dcopf', ['DCOPF']),
    ('ed', ['ED']),
    ('rted', ['RTED', 'RTED2']),
    # ('uc', ['UC']),
    ('dopf', ['LDOPF', 'LDOPF2']),
])

class_names = list_flatten(list(all_routines.values()))
routine_cli = OrderedDict([(item.lower(), item) for item in class_names])
