"""
Dispatch routines.
"""

from collections import OrderedDict
from andes.utils.func import list_flatten

all_routines = OrderedDict([('pflow', ['DCPF', 'PF']),
                            ])

class_names = list_flatten(list(all_routines.values()))
routine_cli = OrderedDict([(item.lower(), item) for item in class_names])

all_models = OrderedDict([('DCPF', ['Bus', 'PV', 'Slack']),
                          ('PF', ['Bus', 'PV', 'Slack']),
                          ])
