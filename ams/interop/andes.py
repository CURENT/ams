"""
Interface with ANDES
"""

import os
import logging
from collections import OrderedDict, Counter

from andes.shared import pd, np
from andes.utils.misc import elapsed
from andes.system import System as andes_System
from andes.interop.pandapower import make_link_table  # NOQA

from ams.io import input_formats
from ams.models.group import StaticGen
from ams.models.static import PV, Slack

logger = logging.getLogger(__name__)


# Models used in ANDES PFlow
pflow_dict = OrderedDict([
    ('Bus', ['idx', 'u', 'name',
             'Vn', 'vmax', 'vmin',
             'v0', 'a0', 'xcoord', 'ycoord',
             'area', 'zone', 'owner']),
    ('PQ', ['idx', 'u', 'name',
            'bus', 'Vn', 'p0', 'q0',
            'vmax', 'vmin', 'owner']),
    ('PV', ['idx', 'u', 'name', 'Sn',
            'Vn', 'bus', 'busr', 'p0', 'q0',
            'pmax', 'pmin', 'qmax', 'qmin',
            'v0', 'vmax', 'vmin', 'ra', 'xs']),
    ('Slack', ['idx', 'u', 'name', 'Sn',
               'Vn', 'bus', 'busr', 'p0', 'q0',
               'pmax', 'pmin', 'qmax', 'qmin',
               'v0', 'vmax', 'vmin', 'ra', 'xs',
               'a0']),
    ('Shunt', ['idx', 'u', 'name', 'Sn',
               'Vn', 'g', 'b', 'fn']),
    ('Line', ['idx', 'u', 'name',
              'bus1', 'bus2', 'Sn',
              'fn', 'Vn1', 'Vn2',
              'r', 'x', 'b', 'g', 'b1', 'g1', 'b2', 'g2',
              'trans', 'tap', 'phi',
              'rate_a', 'rate_b', 'rate_c',
              'owner', 'xcoord', 'ycoord']),
    ('Area', ['idx', 'u', 'name']),
])

# dict for guessing dynamic models given its idx
idx_guess = {'rego': 'RenGovernor',
             'ree': 'RenExciter',
             'rea': 'RenAerodynamics',
             'rep': 'RenPitch',
             'busf': 'BusFreq',
             'zone': 'Region',
             'gen': 'StaticGen',
             'pq': 'PQ', }


def to_andes(system, setup=False, addfile=None,
             **kwargs):
    """
    Convert the AMS system to an ANDES system.

    This function is wrapped as the ``System`` class method ``to_andes()``.
    Using the file conversion ``to_andes()`` will automatically
    link the AMS system instance to the converted ANDES system instance
    in the AMS system attribute ``dyn``.

    It should be noted that detailed dynamic simualtion requires extra
    dynamic models to be added to the ANDES system, which can be passed
    through the ``addfile`` argument.

    Parameters
    ----------
    system : System
        The AMS system to be converted to ANDES format.
    setup : bool, optional
        Whether to call `setup()` after the conversion. Default is True.
    addfile : str, optional
        The additional file to be converted to ANDES dynamic mdoels.
    **kwargs : dict
        Keyword arguments to be passed to `andes.system.System`.

    Returns
    -------
    adsys : andes.system.System
        The converted ANDES system.

    Examples
    --------
    >>> import ams
    >>> import andes
    >>> sp = ams.load(ams.get_case('ieee14/ieee14_rted.xlsx'), setup=True)
    >>> sa = sp.to_andes(setup=False,
    ...                  addfile=andes.get_case('ieee14/ieee14_wt3.xlsx'),
    ...                  overwrite=True, no_keep=True, no_output=True)

    Notes
    -----
    1. Power flow models in the addfile will be skipped and only dynamic models will be used.
    2. The addfile format is guessed based on the file extension. Currently only ``xlsx`` is supported.
    3. Index in the addfile is automatically adjusted when necessary.
    """
    t0, _ = elapsed()

    adsys = andes_System()

    for mdl_name, mdl_cols in pflow_dict.items():
        mdl = getattr(system, mdl_name)
        for row in mdl.cache.df_in[mdl_cols].to_dict(orient='records'):
            adsys.add(mdl_name, row)

    _, s = elapsed(t0)

    # additonal file for dynamic models
    if addfile:
        t_add, _ = elapsed()

        # --- parse addfile ---
        adsys = parse_addfile(adsys=adsys, amsys=system, addfile=addfile)

        _, s_add = elapsed(t_add)
        logger.info('Addfile parsed in %s.', s_add)

    logger.info(f'System converted to ANDES in {s}.')

    # finalize
    system.dyn = Dynamic(amsys=system, adsys=adsys)
    if setup:
        adsys.setup()
        system.dyn.link_andes(adsys=adsys)
    else:
        msg = 'System has not been linked to ANDES. Call ``dyn.link_andes(adsys=sa)`` after setup ANDES system.'
        logger.warning(msg)
    return adsys


def parse_addfile(adsys, amsys, addfile):
    """
    Parse the addfile for ANDES dynamic file.

    Parameters
    ----------
    adsys : andes.system.System
        The ANDES system instance.
    amsys : ams.system.System
        The AMS system instance.
    addfile : str
        The additional file to be converted to ANDES dynamic mdoels.

    Returns
    -------
    adsys : andes.system.System
        The ANDES system instance with dynamic models added.
    """
    # guess addfile format
    add_format = None
    _, add_ext = os.path.splitext(addfile)
    for key, val in input_formats.items():
        if add_ext[1:] in val:
            add_format = key
            logger.debug('Addfile format guessed as %s.', key)
            break

    if key != 'xlsx':
        logger.error('Addfile format "%s" is not supported yet.', add_format)
        # FIXME: xlsx input file with dyr addfile result into KeyError: 'Toggle'
        # add_parser = importlib.import_module('andes.io.' + add_format)
        # if not add_parser.read_add(system, addfile):
        #     logger.error('Error parsing addfile "%s" with %s parser.', addfile, add_format)
        return adsys

    # Try parsing the addfile
    logger.info('Parsing additional file "%s"...', addfile)
    # FIXME: find a better way to handle this, e.g. REGCV1, or ESD1

    reader = pd.ExcelFile(addfile)

    pflow_mdl = list(pflow_dict.keys())

    pflow_mdls_overlap = []
    for mdl_name in pflow_dict.keys():
        if mdl_name in reader.sheet_names:
            pflow_mdls_overlap.append(mdl_name)

    if len(pflow_mdls_overlap) > 0:
        msg = 'Following PFlow models in addfile will be skipped: '
        msg += ', '.join(pflow_mdls_overlap)
        logger.warning(msg)

    pflow_df_models = pd.read_excel(addfile,
                                    sheet_name=pflow_mdl,
                                    index_col=0,
                                    engine='openpyxl',
                                    )
    # drop rows that all nan
    for name, df in pflow_df_models.items():
        df.dropna(axis=0, how='all', inplace=True)

    # collect idx_map if difference exists
    idx_map = OrderedDict([])
    for name, df in pflow_df_models.items():
        am_idx = amsys.models[name].idx.v
        ad_idx = df['idx'].values
        if set(am_idx) != set(ad_idx):
            idx_map[name] = dict(zip(ad_idx, am_idx))

    # --- dynamic models to be added ---
    mdl_to_keep = list(set(reader.sheet_names) - set(pflow_mdl))
    mdl_to_keep.sort(key=str.lower)
    df_models = pd.read_excel(addfile,
                              sheet_name=mdl_to_keep,
                              index_col=0,
                              engine='openpyxl',
                              )

    # adjust models index
    for name, df in df_models.items():
        try:
            mdl = adsys.models[name]
        except KeyError:
            mdl = adsys.model_aliases[name]
        if len(mdl.idx_params) == 0:  # skip if no idx_params
            continue
        for idxn, idxp in mdl.idx_params.items():
            if idxp.model is None:  # make a guess if no model is specified
                mdl_guess = idxn.capitalize()
                if mdl_guess not in adsys.models.keys():
                    try:
                        mdl_guess = idx_guess[idxp.name]
                    except KeyError:  # set the most frequent string as the model name
                        split_list = []
                        for item in df[idxn].values:
                            if item is None or np.nan:
                                continue
                            try:
                                split_list.append(item.split('_'))
                                # Flatten the nested list and filter non-numerical strings
                                flattened_list = [item for sublist in split_list for item in sublist
                                                  if not isinstance(item, int)]
                                # Count the occurrences of non-numerical strings
                                string_counter = Counter(flattened_list)
                                # Find the most common non-numerical string
                                mdl_guess = string_counter.most_common(1)[0][0]
                            except AttributeError:
                                logger.error(f'Failed to parse IdxParam {name}.{idxn}.')
                                continue
            else:
                mdl_guess = idxp.model
            if mdl_guess in adsys.groups.keys():
                grp_idx = {}
                for mname, mdl in adsys.groups[mdl_guess].models.items():
                    # add group index to index map
                    if mname in idx_map.keys():
                        grp_idx.update(idx_map[mname])
                if len(grp_idx) == 0:
                    continue  # no index consistency issue, skip
                idx_map[mdl_guess] = grp_idx
            if mdl_guess not in idx_map.keys():
                continue  # no index consistency issue, skip
            else:
                logger.debug(f'Replace map for {mdl_guess} is {idx_map[mdl_guess]}')
                df[idxn] = df[idxn].replace(idx_map[mdl_guess])
                logger.debug(f'Adjust {idxp.class_name} <{name}.{idxp.name}>')

    # add dynamic models
    for name, df in df_models.items():
        # drop rows that all nan
        df.dropna(axis=0, how='all', inplace=True)
        # if the dynamic model also exists in AMS, use AMS parameters for overlap
        if name in amsys.models.keys():
            if df.shape[0] != amsys.models[name].n:
                msg = f'<{name}> has different rows in AMS and ANDES.'
                logger.warning(msg)
            am_params = set(amsys.models[name].params.keys())
            ad_params = set(df.columns)
            overlap_params = list(am_params.intersection(ad_params))
            ad_rest_params = list(ad_params - am_params) + ['idx']
            try:
                overlap_params.remove('Sn')
            except Exception:
                pass
            msg = f'Following ANDES parameters in <{name}> are overwriten: '
            msg += ', '.join(overlap_params)
            logger.debug(msg)
            tmp = amsys.models[name].cache.df_in[overlap_params]
            df = pd.merge(left=tmp, right=df[ad_rest_params],
                          on='idx', how='left')
        for row in df.to_dict(orient='records'):
            adsys.add(name, row)

    # --- adjust SynGen Vn with Bus Vn ---
    # NOTE: RenGen and DG have no Vn, so no need to adjust
    syg_idx = []
    for _, syg in adsys.SynGen.models.items():
        if syg.n > 0:
            syg_idx += syg.idx.v
    syg_bus_idx = adsys.SynGen.get(src='bus', attr='v', idx=syg_idx)
    syg_bus_vn = adsys.Bus.get(src='Vn', idx=syg_bus_idx)
    adsys.SynGen.set(src='Vn', attr='v', idx=syg_idx, value=syg_bus_vn)

    # --- for debugging ---
    adsys.df_in = df_models

    return adsys


class Dynamic:
    """
    ANDES interface class.

    Parameters
    ----------
    amsys : AMS.system.System
        The AMS system.
    adsys : ANDES.system.System
        The ANDES system.

    Attributes
    ----------
    link : pandas.DataFrame
        The ANDES system link table.

    Notes
    -----
    1. Using the file conversion ``to_andes()`` will automatically
       link the AMS system to the converted ANDES system in the
       attribute ``dyn``.

    Examples
    --------
    >>> import ams
    >>> import andes
    >>> sp = ams.load(ams.get_case('ieee14/ieee14_rted.xlsx'), setup=True)
    >>> sa = sp.to_andes(setup=True,
    ...                  addfile=andes.get_case('ieee14/ieee14_wt3.xlsx'),
    ...                  overwrite=True, keep=False, no_output=True)
    >>> sp.RTED.run()
    >>> sp.RTED.dc2ac()
    >>> sp.dyn.send()  # send RTED results to ANDES system
    >>> sa.PFlow.run()
    >>> sp.TDS.run()
    >>> sp.dyn.receive()  # receive TDS results from ANDES system
    """

    def __init__(self, amsys=None, adsys=None) -> None:
        self.amsys = amsys  # AMS system
        self.adsys = adsys  # ANDES system

        # TODO: add summary table
        self.link = None  # ANDES system link table

    def link_andes(self, adsys):
        """
        Link the ANDES system to the AMS system.

        Parameters
        ----------
        adsys : ANDES.system.System
            The ANDES system instance.
        """
        self.adsys = adsys
        if self.adsys.is_setup:
            self.make_link()
            logger.warning(f'AMS system {hex(id(self.amsys))} is linked to the ANDES system {hex(id(adsys))}.')
        else:
            msg = 'ANDES system is not setup yet.'
            logger.warning(msg)

    @property
    def is_tds(self):
        """
        Indicator of whether the ANDES system is running a TDS.
        This property will return ``True`` as long as TDS is initialized.

        Check ``adsys.tds.TDS.init()`` for more details.
        """
        return bool(self.adsys.TDS.initialized)

    def require_link(func, **kwargs):
        """
        Wrapper function to check if the link table is available before calling
        ``send()`` and ``receive()``.

        Parameters
        ----------
        adsys : adsys.System.system, optional
            The target ANDES dynamic system instance. If not provided, use the
            linked ANDES system instance (`sp.dyn.adsys`).
        """

        def wrapper(self, **kwargs):
            if self.link is None:
                logger.warning("System has not been linked with dynamic. Unable to sync with ANDES.")
                return False
            else:
                return func(self, **kwargs)
        return wrapper

    def make_link(self):
        """
        Make the link table of the ANDES system.

        A wrapper of `adsys.interop.pandapower.make_link_table`.
        """
        self.link = make_link_table(self.adsys)
        # adjust columns
        cols = ['stg_idx', 'bus_idx',               # static gen
                'syg_idx', 'gov_idx',               # syn gen
                'dg_idx',                           # distributed gen
                'rg_idx',                           # renewable gen
                'gammap', 'gammaq',                 # gamma
                ]
        self.link = self.link[cols]

    def _send_tgr(self, sa, sp):
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

    def _send_dgu(self, sa, sp):
        """
        Send to ANDES the dynamic generator online status.
        """
        # 1) SynGen
        syg_idx = sp.dyn.link['syg_idx'].dropna().tolist()  # SynGen idx
        # corresponding StaticGen idx in ANDES
        stg_syg_idx = sa.SynGen.get(src='gen', attr='v', idx=syg_idx,
                                    allow_none=True, default=None)
        # corresponding StaticGen u in AMS
        stg_u_ams = sp.StaticGen.get(src='u', attr='v', idx=stg_syg_idx,
                                     allow_none=True, default=0)
        stg_u_andes = sa.SynGen.get(src='u', attr='v', idx=syg_idx,
                                    allow_none=True, default=0)
        # 2) DG
        dg_idx = sp.dyn.link['dg_idx'].dropna().tolist()  # DG idx
        # corresponding StaticGen idx in ANDES
        stg_dg_idx = sa.DG.get(src='gen', attr='v', idx=dg_idx,
                               allow_none=True, default=None)
        # corresponding DG u in AMS
        dg_u_ams = sp.StaticGen.get(src='u', attr='v', idx=stg_dg_idx,
                                    allow_none=True, default=0)
        du_u_andes = sa.DG.get(src='u', attr='v', idx=dg_idx,
                               allow_none=True, default=0)
        # 3) RenGen
        rg_idx = sp.dyn.link['rg_idx'].dropna().tolist()  # RenGen idx
        # corresponding StaticGen idx in ANDES
        stg_rg_idx = sa.RenGen.get(src='gen', attr='v', idx=rg_idx,
                                   allow_none=True, default=None)
        # corresponding RenGen u in AMS
        rg_u_ams = sp.StaticGen.get(src='u', attr='v', idx=stg_rg_idx,
                                    allow_none=True, default=0)
        rg_u_andes = sa.RenGen.get(src='u', attr='v', idx=rg_idx,
                                   allow_none=True, default=0)
        # 4) sync results
        cond = (
            not np.array_equal(stg_u_ams, stg_u_andes) or
            not np.array_equal(dg_u_ams, du_u_andes) or
            not np.array_equal(rg_u_ams, rg_u_andes)
        )
        if cond:
            msg = 'ANDES dynamic generator online status should be switched using Toggle!'
            msg += ' Otherwise, unexpected results might occur.'
            raise ValueError(msg)
        # FIXME: below code seems to be unnecessary
        sa.SynGen.set(src='u', attr='v', idx=syg_idx, value=stg_u_ams)
        sa.DG.set(src='u', attr='v', idx=dg_idx, value=dg_u_ams)
        sa.RenGen.set(src='u', attr='v', idx=rg_idx, value=rg_u_ams)
        return True

    @require_link
    def send(self, adsys=None, routine=None):
        """
        Send results of the recent sovled AMS dispatch (``sp.recent``) to the
        target ANDES system.

        Note that converged AC conversion DOES NOT guarantee successful dynamic
        initialization ``TDS.init()``.
        Failed initialization is usually caused by limiter violation.

        Parameters
        ----------
        adsys : adsys.System.system, optional
            The target ANDES dynamic system instance. If not provided, use the
            linked ANDES system isntance (``sp.dyn.adsys``).
        routine : str, optional
            The routine to be sent to ANDES. If None, ``recent`` will be used.
        """
        sa = adsys if adsys is not None else self.adsys
        sp = self.amsys
        # 1. information
        rtn = sp.recent if routine is None else getattr(sp, routine)
        if rtn.exit_code != 0:
            logger.warning(f'{sp.recent.class_name} is not solved optimally.')
            return False
        # NOTE:if DC types, check if results are converted
        if rtn.type != 'ACED':
            if rtn.is_ac:
                logger.debug(f'{rtn.class_name} results has been converted to AC.')
            else:
                msg = f'{rtn.class_name} has not been converted'
                msg += ' to AC, error might occur!'
                logger.error(msg)

        try:
            logger.warning(f'Send {rtn.class_name} results to ANDES <{hex(id(sa))}>')
        except AttributeError:
            logger.warning('No solved routine found. Unable to sync with ANDES.')
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

        # 2. sync to dynamic if initialized
        if self.is_tds:
            logger.warning('TDS is initialized, send to <tds> models.')
            # 1) send Dynamic generator online status
            self._send_dgu(sa=sa, sp=sp)
            # 2) send other results
            is_tgr_set = False
            for mname, pmap in map2.items():
                mdl = getattr(sa, mname)
                for ams_vname, andes_pname in pmap.items():
                    if andes_pname in mdl.vars.keys():
                        logger.warning('Setting values to ANDES var is not allowed!')
                        continue
                    # a. power reference
                    # FIXME: in this implementation, PV and Slack are only sync once, this might
                    # cause problem if PV and Slack are mapped separately
                    if not is_tgr_set and isinstance(mdl, (StaticGen, PV, Slack)):
                        self._send_tgr(sa=sa, sp=sp)
                        is_tgr_set = True
                        logger.warning('Generator power reference are sent to dynamic devices')
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
            return True

        # 3. sync to static if not initialized
        else:
            if sa.PFlow.exec_time > 0:
                msg = 'ANDES PFlow has been run, please re-run it'
                msg += ' before running TDS to avoid failed TDS initialization'
                logger.info(msg)
            logger.warning('TDS is not initialized, send to <pflow> models')
            for mdl_name, pmap in map2.items():
                for ams_vname, andes_pname in map2[mdl_name].items():
                    # a. voltage reference
                    if ams_vname in ['qg']:
                        logger.warning('Setting `qg` to ANDES dynamic does not take effect.')
                    # b. others, if any
                    ams_var = getattr(sp.recent, ams_vname)
                    # NOTE: here we stick with ANDES device idx
                    idx = ams_var.get_idx()
                    logger.debug(f'Send <{ams_vname}> to {mdl_name} {andes_pname}')
                    v_ams = sp.recent.get(src=ams_vname, attr='v', idx=idx)
                    mdl_andes = getattr(sa, mdl_name)
                    mdl_andes.set(src=andes_pname, idx=idx, attr='v', value=v_ams)
                    logger.debug(f'Send <{ams_vname}> to {mdl_andes.class_name} {andes_pname}')
            # correct StaticGen v0 with Bus v0 if any
            try:
                rtn.map2['Bus']['vBus'] == 'v0'
                logger.warning('Adjust ANDES StaticGen v0 with Bus v0')
                stg_idx = sa.PV.idx.v + sa.Slack.idx.v
                stg_bus = sa.PV.bus.v + sa.Slack.bus.v
                vBus = sp.Bus.get(src='v0', attr='v', idx=stg_bus)
                sa.StaticGen.set(src='v0', attr='v', idx=stg_idx, value=vBus)
            except KeyError:
                msg = 'Skip adjusting ANDES StaticGen v0 because Bus v0 is not found in map2'
                logger.warning(msg)
            return True

    @require_link
    def receive(self, adsys=None):
        """
        Receive the results from the target ANDES system.

        Parameters
        ----------
        adsys : adsys.System.system, optional
            The target ANDES dynamic system instance. If not provided, use the
            linked ANDES system isntance (``sp.dyn.adsys``).
        """
        sa = adsys if adsys is not None else self.adsys
        sp = self.amsys
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
            for mname, mdl in self.amsys.models.items():
                # NOTE: skip models without idx: ``Summary``
                if not hasattr(mdl, 'idx'):
                    continue
                if mdl.n == 0:
                    continue
                # a. dynamic generator online status
                if not is_dgu_set and mdl.group in ['StaticGen']:
                    self._receive_dgu(sa=sa, sp=sp)
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
                        Pe, idx = self._receive_pe(sa=sa, sp=sp)
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
                # NOTE: skip models without idx: ``Summary``
                if not hasattr(mdl, 'idx'):
                    continue
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

    def _receive_pe(self, sa, sp):
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

    def _receive_dgu(self, sa, sp):
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
