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
    ('ed', ['ED', 'ED2']),
    ('rted', ['RTED', 'RTED2']),
    ('uc', ['UC']),
    ('dopf', ['LDOPF', 'LDOPF2']),
])

class_names = list_flatten(list(all_routines.values()))
routine_cli = OrderedDict([(item.lower(), item) for item in class_names])
