"""
EV model.
"""

from andes.core.param import NumParam

from ams.core.model import Model

class EV(Model):
    """
    EV aggregation model at transmission level.

    It is expected to be used in conjunction with the `EV1` or `EV2` model
    in ANDES.

    Reference:

    [1] J. Wang et al., "Electric Vehicles Charging Time Constrained Deliverable Provision of Secondary 
    Frequency Regulation," in IEEE Transactions on Smart Grid, doi: 10.1109/TSG.2024.3356948.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.group = 'DG'
        
