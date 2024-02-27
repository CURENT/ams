from andes.models.shunt.shunt import ShuntData

from ams.core.model import Model


class Shunt(ShuntData, Model):
    """
    Phasor-domain shunt compensator Model.
    """

    def __init__(self, system=None, config=None):
        ShuntData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'StaticShunt'
