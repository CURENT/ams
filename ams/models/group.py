import logging

from andes.models.group import GroupBase as adGroupBase
from andes.core.service import BackRef

logger = logging.getLogger(__name__)


class GroupBase(adGroupBase):
    """
    Base class for groups.

    Add common_vars and common_params to the group class.

    This class is revised from ``andes.models.group.GroupBase``.
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('idx',))

    def get_idx(self):
        """
        Return the value of group idx sorted in a human-readable style.

        Notes
        -----
        This function sorts the idx values using a custom sorting key,
        which handles varying length strings with letters and numbers.
        """
        all_idx = [mdl.idx.v for mdl in self.models.values()]
        flat_list = [item for sublist in all_idx for item in sublist]
        return flat_list

    def _check_src(self, src: str):
        """
        Helper function for checking if ``src`` is a shared field.

        The requirement is not strictly enforced and is only for debugging purposed.

        Disable debug logging in dispath modeling.
        """
        if src not in self.common_vars + self.common_params:
            logger.debug(f'Group <{self.class_name}> does not share property <{src}>.')

    def __repr__(self):
        dev_text = 'device' if self.n == 1 else 'devices'
        return f'{self.class_name} ({self.n} {dev_text}) at {hex(id(self))}'

    def __setattr__(self, key, value):
        if hasattr(value, 'owner'):
            if value.owner is None:
                value.owner = self
        if hasattr(value, 'name'):
            if value.name is None:
                value.name = key

        if isinstance(value, BackRef):
            self.services_ref[key] = value

        super().__setattr__(key, value)

    def set_backref(self, name, from_idx, to_idx):
        """
        Set idxes to ``BackRef``, and set them to models.
        """

        uid = self.idx2uid(to_idx)
        self.services_ref[name].v[uid].append(from_idx)

        model = self.idx2model(to_idx)
        model.set_backref(name, from_idx, to_idx)


class Undefined(GroupBase):
    """
    The undefined group. Holds models with no ``group``.
    """

    def __init__(self):
        super().__init__()


class ACTopology(GroupBase):
    def __init__(self):
        super().__init__()
        self.common_vars.extend(('a', 'v'))


class RenGen(GroupBase):
    """
    Renewable generator (converter) group.

    See ANDES Documentation SynGen here for the notes on replacing StaticGen and setting the power
    ratio parameters.

    Reference:

    [1] ANDES Documentation, RenGen, [Online]

    Available:

    https://docs.andes.app/en/latest/groupdoc/RenGen.html#rengen
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('bus', 'gen', 'Sn', 'q0'))
        self.common_vars.extend(('Pe', 'Qe'))


class VSG(GroupBase):
    """
    Renewable generator with virtual synchronous generator (VSG) control group.

    Note that this is a group separate from ``RenGen`` for VSG scheduling study.
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('bus', 'gen', 'Sn'))
        self.common_vars.extend(('Pe', 'Qe'))


class DG(GroupBase):
    """
    Distributed generation (small-scale).

    See ANDES Documentation SynGen here for the notes on replacing StaticGen and setting the power
    ratio parameters.

    Reference:

    [1] ANDES Documentation, SynGen, [Online]

    Available:

    https://docs.andes.app/en/latest/groupdoc/SynGen.html#syngen
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('bus', 'fn'))


class Cost(GroupBase):
    def __init__(self):
        super().__init__()
        self.common_params.extend(('gen',))


class Collection(GroupBase):
    """
    Collection of topology models
    """

    def __init__(self):
        super().__init__()


class Horizon(GroupBase):
    """
    Time horizon group.
    """

    def __init__(self):
        super().__init__()


class Reserve(GroupBase):
    def __init__(self):
        super().__init__()
        self.common_params.extend(('zone',))


class StaticGen(GroupBase):
    """
    Static Generator Group.

    The generator types and fuel types are referenced from MATPOWER.

    Generator Types
    ---------------
    The following codes represent the types of generators:
        - BA  : Energy Storage, Battery
        - CE  : Energy Storage, Compressed Air
        - CP  : Energy Storage, Concentrated Solar Power
        - FW  : Energy Storage, Flywheel
        - PS  : Hydraulic Turbine, Reversible (pumped storage)
        - ES  : Energy Storage, Other
        - ST  : Steam Turbine (includes nuclear, geothermal, and solar steam)
        - GT  : Combustion (Gas) Turbine
        - IC  : Internal Combustion Engine (diesel, piston, reciprocating)
        - CA  : Combined Cycle Steam Part
        - CT  : Combined Cycle Combustion Turbine Part
        - CS  : Combined Cycle Single Shaft
        - CC  : Combined Cycle Total Unit
        - HA  : Hydrokinetic, Axial Flow Turbine
        - HB  : Hydrokinetic, Wave Buoy
        - HK  : Hydrokinetic, Other
        - HY  : Hydroelectric Turbine
        - BT  : Turbines Used in a Binary Cycle
        - PV  : Photovoltaic
        - WT  : Wind Turbine, Onshore
        - WS  : Wind Turbine, Offshore
        - FC  : Fuel Cell
        - OT  : Other
        - UN  : Unknown
        - JE  : Jet Engine
        - NB  : ST - Boiling Water Nuclear Reactor
        - NG  : ST - Graphite Nuclear Reactor
        - NH  : ST - High Temperature Gas Nuclear Reactor
        - NP  : ST - Pressurized Water Nuclear Reactor
        - IT  : Internal Combustion Turbo Charged
        - SC  : Synchronous Condenser
        - DC  : DC ties
        - MP  : Motor/Pump
        - W1  : Wind Turbine, Type 1
        - W2  : Wind Turbine, Type 2
        - W3  : Wind Turbine, Type 3
        - W4  : Wind Turbine, Type 4
        - SV  : Static Var Compensator
        - DL  : Dispatchable Load

    Fuel Types
    ----------
    The following codes represent the fuel types:
        - biomass     : Biomass
        - coal        : Coal
        - dfo         : Distillate Fuel Oil
        - geothermal  : Geothermal
        - hydro       : Hydro
        - hydrops     : Hydro Pumped Storage
        - jetfuel     : Jet Fuel
        - lng         : Liquefied Natural Gas
        - ng          : Natural Gas
        - nuclear     : Nuclear
        - oil         : Unspecified Oil
        - refuse      : Refuse, Municipal Solid Waste
        - rfo         : Residual Fuel Oil
        - solar       : Solar
        - syncgen     : Synchronous Condenser
        - wasteheat   : Waste Heat
        - wind        : Wind
        - wood        : Wood or Wood Waste
        - other       : Other
        - unknown     : Unknown
        - dl          : Dispatchable Load
        - ess         : Energy Storage System

    Notes
    -----
    For co-simulation with ANDES, refer to the `ANDES StaticGen Documentation
    <https://docs.andes.app/en/latest/groupdoc/StaticGen.html#staticgen>`_ for
    replacing static generators with dynamic generators.
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('bus', 'Sn', 'Vn', 'p0', 'q0', 'ra', 'xs', 'subidx',
                                   'pmax', 'pmin', 'pg0', 'ctrl', 'R10', 'td1', 'td2',
                                   'area', 'zone', 'gentype', 'genfuel'))
        self.common_vars.extend(('p', 'q'))


class ACLine(GroupBase):
    def __init__(self):
        super().__init__()
        self.common_params.extend(('bus1', 'bus2', 'r', 'x'))


class ACShort(GroupBase):
    def __init__(self):
        super(ACShort, self).__init__()
        self.common_params.extend(('bus1', 'bus2'))


class StaticLoad(GroupBase):
    """
    Static load group.
    """

    def __init__(self):
        super().__init__()
        self.common_params.extend(('bus', 'p0', 'q0', 'ctrl', 'area', 'zone'))


class StaticShunt(GroupBase):
    """
    Static shunt compensator group.
    """
    pass


class Information(GroupBase):
    """
    Group for information container models.
    """

    def __init__(self):
        GroupBase.__init__(self)
        self.common_params = []
