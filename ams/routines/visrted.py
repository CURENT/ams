from ams.routines.rted import rted

class visrted(rted):
    """Virtual inertia scheduling RTED"""

    def __init__(self) -> None:
        rted.__init__(self)
        pass
