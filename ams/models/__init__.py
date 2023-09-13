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
    ('distributed', ['ESD1']),
    ('renewable', ['REGCV1']),
    ('area', ['Area']),
    ('region', ['Region']),
    ('reserve', ['SFR']),
    ('cost', ['GCost', 'SFRCost', 'PFRCost']),
    ('timeslot', ['TimeSlot', 'EDTSlot', 'UCTSlot']),
])

file_classes = ams_file_classes
