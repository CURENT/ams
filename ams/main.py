"""
Main entry point for the AMS CLI and scripting interfaces.
"""

from ams.system import System


def load(case, **kwargs):
    """
    Load a case and return an AMS system.
    """
    system = System(case, **kwargs)
    return system
