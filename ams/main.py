"""
Main entry point for the AMS CLI and scripting interfaces.
"""

from ams.system import System


def load(case):
    """
    Load a case and set up a system without running routine.
    Return a system.
    """
    system = System(case)
    return system
