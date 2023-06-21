"""
Interface with ANDES
"""

import os
import logging

from andes.shared import pd, np
from andes.utils.misc import elapsed
from andes import load as andes_load
from andes.interop.pandapower import make_link_table

from ams.io import input_formats, xlsx, json
from ams.models.group import StaticGen
from ams.models.static import PV, Slack

logger = logging.getLogger(__name__)


def to_andes(system, setup=True, addfile=None,
             overwrite=None, keep=False,
             **kwargs):
    """
    Convert the AMS system to an ANDES system.

    This function is wrapped as the ``System`` class method ``to_andes()``.
    Using the file conversion ``sp.to_andes()`` will automatically
    link the AMS system instance to the converted ANDES system instance
    in the AMS system attribute ``sp.dyn``.

    Parameters
    ----------
    system : System
        The AMS system to be converted to ANDES format.
    setup : bool, optional
        Whether to call `setup()` after the conversion. Default is True.
    addfile : str, optional
        The additional file to be converted to ANDES dynamic mdoels.
    overwrite : bool, optional
        Whether to overwrite the existing file.
    keep : bool, optional
        Whether to keep the converted file.
    **kwargs : dict
        Keyword arguments to be passed to `andes.system.System`.

    Returns
    -------
    andes : andes.system.System
        The converted ANDES system.

    Examples
    --------
    >>> import ams
    >>> import andes
    >>> sp = ams.load(ams.get_case('ieee14/ieee14_rted.xlsx'), setup=True)
    >>> sa = sp.to_andes(setup=False,
    ...                  addfile=andes.get_case('ieee14/ieee14_wt3.xlsx'),
    ...                  overwrite=True, keep=False, no_output=True)
    """
    t0, _ = elapsed()
    andes_file = system.files.name + '.json'

    json.write(system, andes_file,
               overwrite=overwrite,
               to_andes=True,
               )

    _, s = elapsed(t0)
    logger.info(f'System convert to ANDES in {s}, save to "{andes_file}".')

    andes = andes_load(andes_file, setup=False, **kwargs)

    if not keep:
        logger.info(f'Converted file is removed. Set "keep = True" to keep it.')
        os.remove(andes_file)

    # additonal file for dynamic simulation
    add_format = None
    if addfile:
        t, _ = elapsed()

        # guess addfile format
        _, add_ext = os.path.splitext(addfile)
        for key, val in input_formats.items():
            if add_ext[1:] in val:
                add_format = key
                logger.debug('Addfile format guessed as %s.', key)
                break

        # Try parsing the addfile
        logger.info('Parsing additional file "%s"...', addfile)
        # FIXME: hard-coded list of power flow models
        pflow_mdl = ['Bus', 'PQ', 'PV', 'Slack', 'Shunt', 'Line', 'Area']
        if key == 'xlsx':
            # TODO: check if power flow devices exist in the addfile
            reader = pd.ExcelFile(addfile)

            common_elements = set(reader.sheet_names) & set(pflow_mdl)
            if common_elements:
                logger.warning('Power flow models exist in the addfile. Only dynamic models will be kept.')
            mdl_to_keep = list(set(reader.sheet_names) - set(pflow_mdl))
            df_models = pd.read_excel(addfile,
                                      sheet_name=mdl_to_keep,
                                      index_col=0,
                                      engine='openpyxl',
                                      )

            for name, df in df_models.items():
                # drop rows that all nan
                df.dropna(axis=0, how='all', inplace=True)
                for row in df.to_dict(orient='records'):
                    andes.add(name, row)

            # --- for debugging ---
            andes.df_in = df_models

        else:
            logger.warning('Addfile format "%s" is not supported yet.', add_format)
            # FIXME: xlsx input file with dyr addfile result into KeyError: 'Toggle'
            # add_parser = importlib.import_module('andes.io.' + add_format)
            # if not add_parser.read_add(system, addfile):
            #     logger.error('Error parsing addfile "%s" with %s parser.', addfile, add_format)

        # --- consistency check ---
        stg_idx = andes.StaticGen.find_idx(keys='bus',
                                           values=andes.Bus.idx.v,
                                           allow_none=True, default=None)
        stg_idx = [x for x in stg_idx if x is not None]

        syg_idx = andes.SynGen.find_idx(keys='bus',
                                        values=andes.Bus.idx.v,
                                        allow_none=True,
                                        default=None)
        syg_idx = [x for x in syg_idx if x is not None]

        rg_idx = andes.RenGen.find_idx(keys='bus',
                                       values=andes.Bus.idx.v,
                                       allow_none=True, default=None)
        rg_idx = [x for x in rg_idx if x is not None]

        dg_idx = andes.DG.find_idx(keys='bus',
                                   values=andes.Bus.idx.v,
                                   allow_none=True, default=None)
        dg_idx = [x for x in dg_idx if x is not None]

        syg_bus = andes.SynGen.get(src='bus', idx=syg_idx, attr='v')
        rg_bus = andes.RenGen.get(src='bus', idx=rg_idx, attr='v')
        dg_bus = andes.DG.get(src='bus', idx=dg_idx, attr='v')

        # SynGen gen with StaticGen idx
        if any(item not in stg_idx for item in syg_idx):
            logger.debug("Correct SynGen idx to match StaticGen.")
            syg_stg = andes.StaticGen.find_idx(keys='bus',
                                               values=syg_bus,
                                               allow_none=True, default=None)
            andes.SynGen.set(src='gen', idx=syg_idx, attr='v', value=syg_stg)
        if any(item not in stg_idx for item in rg_idx):
            logger.debug("Correct RenGen idx to match StaticGen.")
            rg_stg = andes.StaticGen.find_idx(keys='bus',
                                              values=rg_bus,
                                              allow_none=True, default=None)
            andes.RenGen.set(src='gen', idx=rg_idx, attr='v', value=rg_stg)
        if any(item not in stg_idx for item in dg_idx):
            logger.debug("Correct DG idx to match StaticGen.")
            dg_stg = andes.StaticGen.find_idx(keys='bus', values=dg_bus,
                                              allow_none=True, default=None)
            andes.DG.set(src='gen', idx=dg_idx, attr='v', value=dg_stg)

        # SynGen Vn with Bus Vn
        syg_vn = andes.SynGen.get(src='Vn', idx=syg_idx, attr='v')
        bus_vn = andes.Bus.get(src='Vn', idx=syg_bus, attr='v')
        if not np.equal(syg_vn, bus_vn).all():
            andes.SynGen.set(src='Vn', idx=syg_idx, attr='v', value=bus_vn)
            logger.warning('Correct SynGen Vn to match Bus Vn.')

        # NOTE: RenGen and DG do not have Vn

        _, s = elapsed(t)
        logger.info('Addfile parsed in %s.', s)

    if setup:
        andes.setup()

    t0, _ = elapsed()
    system.dyn = Dynamic(ams=system, andes=andes)
    system.dyn.link_andes(andes=andes)
    _, s = elapsed(t0)
    logger.info(f'AMS system {hex(id(system))} link to ANDES system {hex(id(andes))} in %s.', s)
    return andes


class Dynamic:
    """
    ANDES interface class.

    Parameters
    ----------
    ams : AMS system
        The AMS system.
    andes : ANDES system
        The ANDES system.

    Attributes
    ----------
    link : pandas.DataFrame
        The ANDES system link table.

    Notes
    -----
    1. Using the file conversion ``sp.to_andes()`` will automatically
       link the AMS system to the converted ANDES system in the
       attribute ``sp.dyn``.

    Examples
    --------
    >>> import ams
    >>> import andes
    >>> sp = ams.load(ams.get_case('ieee14/ieee14_rted.xlsx'), setup=True)
    >>> sa = sp.to_andes(setup=True,
    ...                  addfile=andes.get_case('ieee14/ieee14_wt3.xlsx'),
    ...                  overwrite=True, keep=False, no_output=True)
    >>> sp.RTED.run()
    >>> sp.RTED.smooth()
    >>> sp.dyn.send()  # send RTED results to ANDES system
    >>> sa.PFlow.run()
    >>> sp.TDS.run()
    >>> sp.dyn.receive()  # receive TDS results from ANDES system
    """

    def __init__(self, ams=None, andes=None) -> None:
        self.ams = ams  # AMS system
        self.andes = andes  # ANDES system

        # TODO: add summary table
        self.link = None  # ANDES system link table

    def link_andes(self, andes):
        """
        Link the ANDES system to the AMS system.

        Parameters
        ----------
        andes : ANDES.system.System
            The ANDES system instance.
        """
        self.andes = andes
        self.make_link()

    @property
    def is_tds(self):
        """
        Indicator of whether the ANDES system is running a TDS.
        This property will return ``True`` as long as TDS is initialized.

        Check ``andes.tds.TDS.init()`` for more details.
        """
        return bool(self.andes.TDS.initialized)

    def make_link(self):
        """
        Make the link table of the ANDES system.

        A wrapper of `andes.interop.pandapower.make_link_table`.
        """
        self.link = make_link_table(self.andes)
        # adjust columns
        cols = ['stg_idx', 'bus_idx',               # static gen
                'syg_idx', 'gov_idx',               # syn gen
                'dg_idx',                           # distributed gen
                'rg_idx',                           # renewable gen
                'gammap', 'gammaq',                 # gamma
                ]
        self.link = self.link[cols]

    def send_tgr(self, sa, sp):
        """
        Sned to generator power refrence.

        Notes
        -----
        1. AGC power reference ``paux`` is not included in this function.
        """
        # 1) TurbineGov
        syg_idx = sp.dyn.link['syg_idx'].dropna().tolist()  # SynGen idx
        # corresponding StaticGen idx in ANDES
        stg_syg_idx = sa.SynGen.get(src='gen', attr='v', idx=syg_idx,
                                    allow_none=True, default=None)
        # corresponding TurbineGov idx in ANDES
        gov_idx = sa.TurbineGov.find_idx(keys='syn', values=syg_idx)
        # corresponding StaticGen pg in AMS
        syg_ams = sp.recent.get(src='pg', attr='v', idx=stg_syg_idx,
                                allow_none=True, default=0)
        # --- check consistency ---
        syg_mask = self.link['syg_idx'].notnull() & self.link['gov_idx'].isnull()
        if syg_mask.any():
            logger.debug('Governor is not complete for SynGen.')
        # --- pref ---
        sa.TurbineGov.set(value=syg_ams, idx=gov_idx,
                          src='pref0', attr='v')

        # --- paux ---
        # TODO: sync paux, using paux0

        # 2) DG
        dg_idx = sp.dyn.link['dg_idx'].dropna().tolist()  # DG idx
        # corresponding StaticGen idx in ANDES
        stg_dg_idx = sa.DG.get(src='gen', attr='v', idx=dg_idx,
                               allow_none=True, default=None,
                               )
        # corresponding StaticGen pg in AMS
        dg_ams = sp.recent.get(src='pg', attr='v', idx=stg_dg_idx,
                               allow_none=True, default=0)
        # --- pref ---
        sa.DG.set(value=dg_ams, idx=dg_idx,
                  src='pref0', attr='v')
        # TODO: paux, using Pext0, this one should be do in other place rather than here

        # 3) RenGen
        # TODO: seems to be unnecessary
        # which models/params are used to control output and auxillary power?

        return True

    def send_dgu(self, sa, sp):
        """
        Send to ANDES the dynamic generator online status.
        """
        # 1) TurbineGov
        syg_idx = sp.dyn.link['syg_idx'].dropna().tolist()  # SynGen idx
        # corresponding StaticGen idx in ANDES
        stg_syg_idx = sa.SynGen.get(src='gen', attr='v', idx=syg_idx,
                                    allow_none=True, default=None)
        # corresponding StaticGen u in AMS
        stg_u_ams = sp.StaticGen.get(src='u', attr='v', idx=stg_syg_idx,
                                     allow_none=True, default=0)
        # set u to dynamic
        sa.SynGen.set(src='u', attr='v', idx=syg_idx, value=stg_u_ams)

        # 2) DG
        dg_idx = sp.dyn.link['dg_idx'].dropna().tolist()  # DG idx
        # corresponding StaticGen idx in ANDES
        stg_dg_idx = sa.DG.get(src='gen', attr='v', idx=dg_idx,
                               allow_none=True, default=None,
                               )
        # corresponding DG u in AMS
        dg_u_ams = sp.StaticGen.get(src='u', attr='v', idx=stg_dg_idx,
                                    allow_none=True, default=0)
        # set u to dynamic
        sa.DG.set(src='u', attr='v', idx=dg_idx, value=dg_u_ams)

        # 3) RenGen
        rg_idx = sp.dyn.link['rg_idx'].dropna().tolist()  # RenGen idx
        # corresponding StaticGen idx in ANDES
        stg_rg_idx = sa.RenGen.get(src='gen', attr='v', idx=rg_idx,
                                   allow_none=True, default=None)
        # corresponding RenGen u in AMS
        rg_u_ams = sp.StaticGen.get(src='u', attr='v', idx=stg_rg_idx,
                                    allow_none=True, default=0)
        # set u to dynamic
        sa.RenGen.set(src='u', attr='v', idx=rg_idx, value=rg_u_ams)

        return True

    def send(self, andes=None):
        """
        Send results of the recent sovled AMS dispatch (``sp.recent``) to the
        target ANDES system.

        Parameters
        ----------
        andes : andes.System.system, optional
            The target ANDES dynamic system instance. If not provided, use the
            linked ANDES system isntance (``sp.dyn.andes``).
        """
        sa = andes if andes is not None else self.andes
        sp = self.ams
        # 1. information
        # NOTE:if DC types, check if results are converted
        if sp.recent.type != 'ACED':
            if sp.recent.is_ac:
                logger.debug(f'{sp.recent.class_name} results has been converted to AC.')
            else:
                logger.warning(
                    f'{sp.recent.class_name} has not been converted to AC, error might be introduced in dynamic simulation.')

        try:
            logger.debug(f'Sending {sp.recent.class_name} results to ANDES <{hex(id(sa))}>...')
        except AttributeError:
            logger.warning('No solved AMS routine found. Unable to sync with ANDES.')
            return False

        # mapping dict
        map2 = getattr(sp.recent, 'map2')
        if len(map2) == 0:
            logger.warning(f'Mapping dict "map2" of {sp.recent.class_name} is empty.')
            return True
        # check map2
        pv_slack = 'PV' in map2.keys() and 'SLACK' in map2.keys()
        if pv_slack:
            logger.warning('PV and SLACK are both in map2, this might cause error when mapping.')

        # 2. sync dynamic results if dynamic is initialized
        if self.is_tds:
            logger.debug(f'Sending results to <tds> models...')
            # 1) send models online status
            # TODO:
            is_dgu_set = False
            for mname, mdl in sa.models.items():
                if mdl.n == 0:
                    continue
                # a. dynamic generator online status
                if not is_dgu_set and mdl.group in ['StaticGen']:
                    self.send_dgu(sa=sa, sp=sp)
                    is_dgu_set = True
                    logger.warning('Generator online status are sent to dynamic generators.')
                    continue
                # b. other models online status
                idx = mdl.idx.v
                if mname in sa.models:
                    mdl_andes = getattr(sa, mname)
                    # 1) send models online status
                    u_ams = mdl.get(src='u', idx=idx, attr='v')
                    mdl_andes.set(src='u', idx=idx, attr='v', value=u_ams)
            # 2) send other results
            is_tgr_set = False
            for mname, pmap in map2.items():
                if mname == 'Bus':
                    logger.warning('Dynamic has been initialized, Bus angle and voltage are skipped.')
                    continue
                mdl = getattr(sa, mname)
                for ams_vname, andes_pname in pmap.items():
                    # a. power reference
                    # FIXME: in this implementation, PV and Slack are only sync once, this might
                    # cause problem if PV and Slack are mapped separately
                    if not is_tgr_set and isinstance(mdl, (StaticGen, PV, Slack)):
                        self.send_tgr(sa=sa, sp=sp)
                        is_tgr_set = True
                        logger.warning('Generator power reference are sent to dynamic devices.')
                        continue
                    # c. others, if any
                    ams_var = getattr(sp.recent, ams_vname)
                    v_ams = sp.recent.get(src=ams_vname, attr='v',
                                          idx=ams_var.get_idx())
                    try:
                        mdl.set(src=andes_pname, attr='v',
                                idx=ams_var.get_idx(), value=v_ams)
                    except KeyError:
                        logger.warning(f'Param {andes_pname} not found in ANDES model <{mname}>.')
                        continue
        # 3. sync static results if dynamic is not initialized
        else:
            logger.debug(f'Sending results to <pflow> models...')
            for mname, mdl in sp.models.items():
                if mdl.n == 0:
                    continue
                idx = mdl.idx.v
                if mname in sa.models:
                    mdl_andes = getattr(sa, mname)
                    # 1) send models online status
                    u_ams = mdl.get(src='u', idx=idx, attr='v')
                    mdl_andes.set(src='u', idx=idx, attr='v', value=u_ams)
                    # 2) send other results
                    # NOTE: send power reference to dynamic device
                    for ams_vname, andes_pname in map2[mname].items():
                        # a. voltage reference
                        if ams_vname in ['qg']:
                            logger.warning('Setting `qg` to ANDES dynamic does not take effect.')
                        # b. others, if any
                        ams_var = getattr(sp.recent, ams_vname)
                        v_ams = sp.recent.get(src=ams_vname, attr='v',
                                              idx=ams_var.get_idx())
                        mdl_andes.set(src=andes_pname, idx=idx, attr='v', value=v_ams)
                return True
            return True

    def receive(self, andes=None):
        """
        Receive the results from the target ANDES system.

        Parameters
        ----------
        andes : andes.System.system, optional
            The target ANDES dynamic system instance. If not provided, use the
            linked ANDES system isntance (``sp.dyn.andes``).
        """
        sa = andes if andes is not None else self.andes
        sp = self.ams
        # 1. information
        try:
            rtn_name = sp.recent.class_name
            logger.debug(f'Receiving ANDES <{hex(id(sa))}> results to {rtn_name}.')
        except AttributeError:
            logger.warning('No target AMS routine found. Failed to sync with ANDES.')
            return False

        # mapping dict
        map1 = getattr(sp.recent, 'map1')
        if len(map1) == 0:
            logger.warning(f'Mapping dict "map1" of {sp.recent.class_name} is empty.')
            return True

        # 2. sync dynamic results if dynamic is initialized
        if self.is_tds:
            # TODO: dynamic results
            logger.debug(f'Receiving <tds> results to {sp.recent.class_name}...')
            # 1) receive models online status
            is_dgu_set = False
            for mname, mdl in self.ams.models.items():
                if mdl.n == 0:
                    continue
                # a. dynamic generator online status
                if not is_dgu_set and mdl.group in ['StaticGen']:
                    self.receive_dgu(sa=sa, sp=sp)
                    is_dgu_set = True
                    logger.warning('Generator online status are received from dynamic generators.')
                    continue
                # b. other models online status
                idx = mdl.idx.v
                if mname in sa.models:
                    mdl_andes = getattr(sa, mname)
                    # 1) receive models online status
                    u_andes = mdl_andes.get(src='u', idx=idx, attr='v')
                    mdl.set(src='u', idx=idx, attr='v', value=u_andes)
            # 2) receive other results
            is_pe_set = False
            for mname, pmap in map1.items():
                mdl = getattr(sa, mname)
                for ams_vname, andes_pname in pmap.items():
                    # a. output power
                    if not is_pe_set and andes_pname == 'p' and mname == 'StaticGen':
                        Pe, idx = self.receive_pe(sa=sa, sp=sp)
                        sp.recent.set(src=ams_vname, idx=idx, attr='v', value=Pe)
                        is_pe_set = True
                        logger.warning('Generator output power are received from dynamic generators.')
                        continue
                    # b. others, if any
                    ams_var = getattr(sp.recent, ams_vname)
                    v_ams = sp.recent.get(src=ams_vname, attr='v',
                                          idx=ams_var.get_idx())
                    try:
                        mdl.set(src=andes_pname, attr='v',
                                idx=ams_var.get_idx(), value=v_ams)
                    except KeyError:
                        logger.warning(f'Param {andes_pname} not found in ANDES model <{mname}>.')
                        continue
            return True
        # 3. sync static results if dynamic is not initialized
        else:
            logger.debug(f'Receiving <pflow> results to {sp.recent.class_name}...')
            for mname, mdl in sp.models.items():
                if mdl.n == 0:
                    continue
                # 1) receive models online status
                idx = mdl.idx.v
                if mname in sa.models:
                    mdl_andes = getattr(sa, mname)
                    u_andes = mdl_andes.get(src='u', idx=idx, attr='v')
                    mdl.set(src='u', idx=idx, attr='v', value=u_andes)
                    # update routine variables if any
                    for vname, var in sp.recent.vars.items():
                        if var.src == 'u':
                            sp.recent.set(src=vname, idx=idx, attr='v', value=u_andes)
                        else:
                            continue
                # 2) receive other results
                # NOTE: receive output power to rotuine
                if mname in map1.keys():
                    for ams_vname, andes_pname in map1[mname].items():
                        v_andes = mdl_andes.get(src=andes_pname, idx=idx, attr='v')
                        sp.recent.set(src=ams_vname, idx=idx, attr='v', value=v_andes)
            return True

    def receive_pe(self, sa, sp):
        """
        Get the dynamic generator output power.
        """
        if not self.is_tds:     # sync dynamic device
            logger.warning('Dynamic is not running, receiving Pe is skipped.')
            return True
        # 1) SynGen
        Pe_sg = sa.SynGen.get(idx=sp.dyn.link['syg_idx'].replace(np.NaN, None).to_list(),
                              src='Pe', attr='v',
                              allow_none=True, default=0,)

        # 2) DG
        Ie_dg = sa.DG.get(src='Ipout_y', attr='v',
                          idx=sp.dyn.link['dg_idx'].replace(np.NaN, None).to_list(),
                          allow_none=True, default=0,)
        v_dg = sa.DG.get(src='v', attr='v',
                         idx=sp.dyn.link['dg_idx'].replace(np.NaN, None).to_list(),
                         allow_none=True, default=0,)
        Pe_dg = v_dg * Ie_dg

        # 3) RenGen
        Pe_rg = sa.RenGen.get(idx=sp.dyn.link['rg_idx'].replace(np.NaN, None).to_list(),
                              src='Pe', attr='v',
                              allow_none=True, default=0,)
        Pe = Pe_sg + Pe_dg + Pe_rg
        idx = sp.dyn.link['stg_idx'].replace(np.NaN, None).to_list()
        return Pe, idx

    def receive_dgu(self, sa, sp):
        """
        Get the dynamic generator online status.
        """
        # 1) SynGen
        u_sg = sa.SynGen.get(idx=sp.dyn.link['syg_idx'].replace(np.NaN, None).to_list(),
                             src='u', attr='v',
                             allow_none=True, default=0,)
        # 2) DG
        u_dg = sa.DG.get(idx=sp.dyn.link['dg_idx'].replace(np.NaN, None).to_list(),
                         src='u', attr='v',
                         allow_none=True, default=0,)
        # 3) RenGen
        u_rg = sa.RenGen.get(idx=sp.dyn.link['rg_idx'].replace(np.NaN, None).to_list(),
                             src='u', attr='v',
                             allow_none=True, default=0,)
        # --- sync into AMS ---
        u_dyngen = u_sg + u_rg + u_dg
        sp.StaticGen.set(idx=sp.dyn.link['stg_idx'].to_list(),
                         src='u', attr='v',
                         value=u_dyngen,)
        return True
