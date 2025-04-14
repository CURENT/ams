"""
The package for models used in scheduling modeling.

The file_classes includes the list of file classes and their corresponding classes.
"""


ams_file_classes = list([
    ('info', ['Summary']),
    ('bus', ['Bus']),
    ('static', ['PQ', 'Slack', 'PV']),
    ('shunt', ['Shunt']),
    ('line', ['Line', 'Jumper']),
    ('distributed', ['PVD1', 'ESD1', 'EV1', 'EV2']),
    ('renewable', ['REGCA1', 'REGCV1', 'REGCV2']),
    ('area', ['Area']),
    ('zone', ['Zone']),
    ('reserve', ['SFR', 'SR', 'NSR', 'VSGR']),
    ('cost', ['GCost', 'SFRCost', 'SRCost', 'NSRCost', 'VSGCost']),
    ('cost', ['DCost']),
    ('timeslot', ['EDTSlot', 'UCTSlot']),
])

file_classes = ams_file_classes
