"""
Dispatch routines.
"""

from collections import OrderedDict
from andes.utils.func import list_flatten
from ams.routines.routine import RoutineBase  # NOQA

all_routines = OrderedDict([
    ('dcpf', ['DCPF']),
    ('pflow', ['PFlow']),
    ('cpf', ['CPF']),
    ('acopf', ['ACOPF']),
    ('dcopf', ['DCOPF']),
    ('ed', ['ED', 'EDDG', 'EDES']),
    ('rted', ['RTED', 'RTEDDG', 'RTEDES', 'RTEDVIS']),
    ('uc', ['UC', 'UCDG', 'UCES']),
    ('dopf', ['DOPF', 'DOPFVIS']),
])

class_names = list_flatten(list(all_routines.values()))
routine_cli = OrderedDict([(item.lower(), item) for item in class_names])
