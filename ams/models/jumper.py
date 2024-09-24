"""
Jumper model for connecting two buses without impedance.
"""

from andes.models.line.jumper import JumperData

from ams.core.model import Model


class Jumper(JumperData, Model):
    """
    Jumper is a device to short two buses (merging two buses into one).

    Jumper can connect two buses satisfying one of the following conditions:

    - neither bus is voltage-controlled
    - either bus is voltage-controlled
    - both buses are voltage-controlled, and the voltages are the same.

    If the buses are controlled in different voltages, power flow will
    not solve (as the power flow through the jumper will be infinite).

    In the solutions, the ``p`` and ``q`` are flowing out of bus1
    and flowing into bus2.

    Setting a Jumper's connectivity status ``u`` to zero will disconnect the two
    buses. In the case of a system split, one will need to call
    ``System.connectivity()`` immediately following the split to detect islands.
    """

    def __init__(self, system=None, config=None) -> None:
        JumperData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'ACShort'
