"""
Interface to PyPower
"""
from numpy import array  # NOQA


class ppi():
    """
    PyPower interface class.

    It has the following methods:
    ``from_ppc``: set the PyPower case dict ``ppc``

    Examples
    --------
    >>> spp = pp_system()
    >>> spp.from_ppc(ams.get_case('pypower/case14.py'))
    """

    def __init__(self):
        self.ppc = None
        # 'ppc' for PyPower case, 'mpc' for Matpower case
        # 'ads' for ANDES system, 'ams' for AMS system
        # TODO: this source_type seems not necessary
        self.source_type = None

    def from_ppc(self, case):
        """
        Set the PyPower case dict ``ppc``.

        Parameters
        ----------
        ppc : dict
            The PyPower case dict.
        """
        self.ppc = set_ppc(case)
        self.source_type = 'ppc'
        # TODO: load into a AMS system or something


def set_ppc(case) -> dict:
    """
    Load PyPower case file into a dict.

    Parameters
    ----------
    case : str
        The path to the PyPower case file.

    Returns
    -------
    ppc : dict
        The PyPower case dict.
    """
    exec(open(f"{case}").read())
    # NOTE: the following line is not robust
    func_name = case.split('/')[-1].rstrip('.py')
    ppc = eval(f"{func_name}()")
    source_type = 'ppc'
    return ppc


def to_ppc(sps):
    """
    Convert the AMS system to a PyPower case dict.

    Parameters
    ----------
    sps : ams.system
        The AMS system.
    """
    # TODO: convert the AMS system to a PyPower case dict
    ppc = None

    return ppc
