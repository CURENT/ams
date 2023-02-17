"""
The package for models in AMS.

The file_classes excludes some of the dynamic models in andes.models.file_classes.
"""


andes_file_classes = list([
    ('info', ['Summary']),
    ('bus', ['Bus']),
    ('static', ['PQ', 'PV', 'Slack']),
    ('shunt', ['Shunt']),
    ('line', ['Line']),
    ('area', ['Area']),
])

# TODO: add AMS exclusive models
ams_file_classes = list([
    ('cost', ['GCost']),
])

file_classes = andes_file_classes + ams_file_classes
