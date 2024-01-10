"""
The package for models used in dispatch modeling.

The file_classes includes the list of file classes and their corresponding classes.
"""


ams_file_classes = list([
    ('info', ['Summary']),
    ('bus', ['Bus']),
    ('static', ['PQ', 'PV', 'Slack']),
    ('shunt', ['Shunt']),
    ('line', ['Line']),
    ('distributed', ['PVD1', 'ESD1']),
    ('renewable', ['REGCA1', 'REGCV1', 'REGCV2']),
    ('area', ['Area']),
    ('region', ['Region']),
    ('reserve', ['SFR', 'SR', 'NSR', 'VSGR']),
    ('cost', ['GCost', 'SFRCost', 'SRCost', 'NSRCost', 'VSGCost']),
    ('cost', ['DCost']),
    ('timeslot', ['TimeSlot', 'EDTSlot', 'UCTSlot']),
])

file_classes = ams_file_classes
