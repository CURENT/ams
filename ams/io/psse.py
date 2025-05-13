"""
PSS/E .raw reader for AMS.
This module is the existing module in ``andes.io.psse``.
"""

from andes.io.psse import testlines  # NOQA
from andes.io.psse import read as ad_read


def read(system, file):
    """
    Read PSS/E RAW file v32/v33 formats.

    Revised from ``andes.io.psse.read`` to complete model ``Zone`` when necessary.
    """
    ret = ad_read(system, file)
    # Extract zone data
    zone = system.Bus.zone.v
    zone_map = {}

    # Check if there are zones to process
    # NOTE: sinece `Zone` and `Area` below to group `Collection`, we add
    # numerical Zone idx after the last Area idx.
    if zone:
        n_zone = system.Area.n
        for z in set(zone):
            # Add new zone and update the mapping
            z_new = system.add(
                'Zone',
                param_dict=dict(idx=n_zone + 1, name=f'{n_zone + 1}')
            )
            zone_map[z] = z_new
            n_zone += 1

        # Update the zone values in the system
        system.Bus.zone.v = [zone_map[z] for z in zone]
    return ret
