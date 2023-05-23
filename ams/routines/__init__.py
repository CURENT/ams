"""
Dispatch routines.
"""

from collections import OrderedDict
from andes.utils.func import list_flatten
from ams.routines.routinebase import RoutineData, RoutineModel

all_routines = OrderedDict([
    ('dcpf', ['DCPF']),
    ('pflow', ['PFlow']),
    ('dcopf', ['DCOPF']),
    ('acopf', ['ACOPF']),
])

class_names = list_flatten(list(all_routines.values()))
routine_cli = OrderedDict([(item.lower(), item) for item in class_names])
