"""
System class for power system data and methods
"""

from typing import Dict, Optional, Tuple, Union

import ams.io


class System:
    """System contains routines for dispatch"""

    def __init__(self,
                 case: Optional[str] = None,) -> None:
        ams.io.parse(self)
