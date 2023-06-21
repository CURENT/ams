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
    ('area', ['Area']),
    ('zone', ['Zone']),
    ('reserve', ['SFR']),
    ('cost', ['GCost', 'SFRCost']),
])

file_classes = ams_file_classes
