"""
Scheduling routines.
"""

from collections import OrderedDict
from andes.utils.func import list_flatten

all_routines = OrderedDict([
    ('dcpf', ['DCPF']),
    ('pflow', ['PFlow']),
    ('cpf', ['CPF']),
    ('acopf', ['ACOPF']),
    ('dcopf', ['DCOPF']),
    ('dcopf2', ['DCOPF2']),
    ('ed', ['ED', 'EDDG', 'EDES']),
    ('rted', ['RTED', 'RTEDDG', 'RTEDES', 'RTEDVIS']),
    ('uc', ['UC', 'UCDG', 'UCES']),
    ('dopf', ['DOPF', 'DOPFVIS']),
    ('pflow0', ['PFlow0']),
    ('dcpf0', ['DCPF0']),
])

class_names = list_flatten(list(all_routines.values()))
routine_cli = OrderedDict([(item.lower(), item) for item in class_names])
