"""
Dispatch routines.
"""

from collections import OrderedDict  # NOQA
from andes.utils.func import list_flatten  # NOQA
from ams.routines.routine import RoutineData, RoutineModel  # NOQA

all_routines = OrderedDict([
    ('dcpf', ['DCPF']),
    ('pflow', ['PFlow']),
    ('cpf', ['CPF']),
    ('acopf', ['ACOPF']),
    ('dcopf', ['DCOPF']),
    ('dcopf2', ['DCOPF2']),
    ('ed', ['ED', 'EDES']),
    ('rted', ['RTED', 'RTEDES']),
    ('uc', ['UC', 'UCES']),
    ('dopf', ['DOPF', 'DOPFVIS']),
])

class_names = list_flatten(list(all_routines.values()))
routine_cli = OrderedDict([(item.lower(), item) for item in class_names])
