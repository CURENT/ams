"""
Scheduling routines.
"""

from collections import OrderedDict
from andes.utils.func import list_flatten

all_routines = OrderedDict([
    ('dcpf', ['DCPF']),
    ('pflow', ['PFlow']),
    ('acopf', ['ACOPF']),
    ('dcopf', ['DCOPF']),
    ('dcopf2', ['DCOPF2']),
    ('rted', ['RTED', 'RTEDDG', 'RTEDESP', 'RTEDES', 'RTEDVIS']),
    ('rted2', ['RTED2', 'RTEDDG2', 'RTEDES2']),
    ('ed', ['ED', 'EDDG', 'EDES']),
    ('ed2', ['ED2']),
    ('uc', ['UC', 'UCDG', 'UCES']),
    ('dopf', ['DOPF', 'DOPFVIS']),
    ('pypower', ['DCPF1', 'PFlow1', 'DCOPF1', 'ACOPF1']),
    ('grbopt', ['OPF']),
])

class_names = list_flatten(list(all_routines.values()))
routine_cli = OrderedDict([(item.lower(), item) for item in class_names])
