"""Economic Dispatch"""

from ams.routines.dcopf import dcopf

class ed(dcopf):
    """Economic Dispatch"""

    def __init__(self) -> None:
        dcopf.__init__(self)
