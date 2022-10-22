"""
System class for dispatch
"""
import ams.io

class System:
    """System contains models and routines"""
    pass

    def __init__(self, case) -> None:
        ams.io.parse(self)
        pass

    def load(self, case: str = None) -> None:
        """Load case into AMS"""
        print("Load case:", case)
        pass

    def build(self, routine_name: str = 'ED') -> None:
        """Build the dispatch model"""
        self.ED = ED()
        pass

    def display(self, routine_name: str = None) -> None:
        """Display the dispatch model"""
        pass
