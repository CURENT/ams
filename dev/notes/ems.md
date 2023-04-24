# Energy management system design

## Overview

When using ANDES in the co-simulation, it might be better to have a energy management system (EMS), or named control center, for the transmission grid. The control center can be used to do regular operational actions and contingency responses.

Further, part of the EMS functions can be integrated with AMS, making the dispatch-dynamic co-simulation easier to organize.

The structure will be like follow:

## Design

EMS performs a number of functions, including State Estimation, Contingency Analysis, Optimal Power Flow (OPF), Load Shedding, and Automatic Generation Control (AGC). OPF and AGC can be good starting points given our previous efforts and projects.

The following considerations should be made:

1. Data exchange. The data exchange between ANDES and EMS is real-time level, or at least every second. This feature can be later integrated into the T-D cosim framework.
2. Flexibility. EMS should be able to have a structure to develop other system-level controllers, such as wide-area damping control.
3. Independence. ANDES should be able to run with or without EMS, allowing usability when necessary and no drawback when focusing on dynamic only.

## Implementation

The design of a module parallel to ANDES/AMS system module can avoid making an ANDES/AMS system too bulky.

One system corresponds with one EMS. The EMS should be initialized with a given ANDES/AMS system, or both but with consistency check.

## Pseudo code
```python
class EMS:
    """
    Energy Management System class.

    This class represents an energy management system that can be used for transmission grid control.

    Parameters
    ----------
    dynsys : module, optional
        Dynamic system module.
    dispatchsys : module, optional
        Dispatch system module.
    """
    def __init__(self, dynsys=None, dispatchsys=None):
        pass

class BaseRoutine:
    """
    Base class for EMS routines.

    This class represents the base class for energy management system routines. It can be subclassed to implement
    specific routines such as AGC or OPF.

    Attributes
    ----------
    ansys : module
        ANDES system module.
    amsys : module, optional
        ANDES-MT system module.
    """
    def __init__(self, ansys, amsys=None):
        pass

    def run(self):
        """
        Run the EMS routine.
        """
        pass

class AGC(BaseRoutine):
    """
    Automatic Generation Control routine.

    This class represents an Automatic Generation Control routine for the energy management system.

    Attributes
    ----------
    ansys : module
        ANDES system module.
    amsys : module, optional
        ANDES-MT system module.
    """
    def __init__(self, ansys, amsys=None):
        pass

    def run(self):
        """
        Run the AGC routine.
        """
        pass
```
