"""
The package for models in AMS.

The file_classes excludes some of the dynamic models in andes.models.file_classes.
"""

from andes.models.info import Summary
from andes.models.bus import Bus
from andes.models.line import Line
from andes.models.static import PQ, PV, Slack
from andes.models.shunt import Shunt
from andes.models.area import Area


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
    # ('cost', ['GCost']),
])

file_classes = andes_file_classes + ams_file_classes
