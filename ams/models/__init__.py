"""
The package for models in AMS.

The file_classes excludes some of the dynamic models in andes.models.file_classes.
"""

from andes.models.info import Summary as Summary
from andes.models.bus import Bus as Bus
from andes.models.line import Line as Line
from andes.models.static import PQ as PQ
from andes.models.static import PV as PV
from andes.models.static import Slack as Slack
from andes.models.shunt import Shunt as Shunt
from andes.models.area import Area as Area

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

