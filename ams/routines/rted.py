"""Real time economic dispatch (RTED)"""
from ams.routines.ed import ed

class rted(ed):
    """Real time economic dispatch (RTED)"""

    def __init__(self) -> None:
        ed.__init__(self)
        pass
