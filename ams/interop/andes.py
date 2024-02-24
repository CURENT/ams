"""
Interface with ANDES
"""

import os
import logging
from collections import OrderedDict, Counter

from andes.shared import pd, np
from andes.utils.misc import elapsed
from andes.system import System as andes_System

from ams.io import input_formats

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
               'Vn', 'bus', 'g', 'b', 'fn']),
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

    A preferred dynamic system file to be added has following features:
    1. The file contains both power flow and dynamic models.
    2. The file can run in ANDES natively.
    3. Power flow models are in the same shape as the AMS system.
    4. Dynamic models, if any, are in the same shape as the AMS system.

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
    >>> sp = ams.load(ams.get_case('ieee14/ieee14_uced.xlsx'), setup=True)
    >>> sa = sp.to_andes(setup=False,
    ...                  addfile=andes.get_case('ieee14/ieee14_full.xlsx'),
    ...                  overwrite=True, no_output=True)

    Notes
    -----
    1. Power flow models in the addfile will be skipped and only dynamic models will be used.
    2. The addfile format is guessed based on the file extension. Currently only ``xlsx`` is supported.
    3. Index in the addfile is automatically adjusted when necessary.
    """
    t0, _ = elapsed()

    adsys = andes_System()
    # FIXME: is there a systematic way to do this? Other config might be needed
    adsys.config.freq = system.config.freq
    adsys.config.mva = system.config.mva

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

    # fake FileManaer attributes
    adsys.files = system.files

    logger.info(f'System converted to ANDES in {s}.')

    # finalize
    system.dyn = Dynamic(amsys=system, adsys=adsys)
    system.dyn.link_andes(adsys=adsys)
    if setup:
        adsys.setup()
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

    reader = pd.ExcelFile(addfile)

    pflow_mdl = list(pflow_dict.keys())

    pflow_mdls_overlap = []
    for mdl_name in pflow_dict.keys():
        if mdl_name in reader.sheet_names:
            pflow_mdls_overlap.append(mdl_name)

    if len(pflow_mdls_overlap) > 0:
        msg = 'Following PFlow models in addfile will be overwritten: '
        msg += ', '.join([f'<{mdl}>' for mdl in pflow_mdls_overlap])
        logger.warning(msg)

    pflow_mdl_nonempty = [mdl for mdl in pflow_mdl if amsys.models[mdl].n > 0]
    logger.debug(f"Non-empty PFlow models: {pflow_mdl}")
    pflow_df_models = pd.read_excel(addfile,
                                    sheet_name=pflow_mdl_nonempty,
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
        if len(set(am_idx)) != len(set(ad_idx)):
            msg = f'<{name}> has different number of rows in addfile.'
            logger.warning(msg)
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
        df.replace(['', ' '], np.NaN, inplace=True)  # replace empty string with nan
        df.dropna(axis=0, how='all', inplace=True)
        # if the dynamic model also exists in AMS, use AMS parameters for overlap
        if (name in amsys.models.keys()) and amsys.models[name].n > 0:
            if df.shape[0] != amsys.models[name].n:
                msg = f'<{name}> has different number of rows in addfile.'
                logger.warning(msg)
            am_params = set(amsys.models[name].params.keys())
            ad_params = set(df.columns)
            overlap_params = list(am_params.intersection(ad_params))
            ad_rest_params = list(ad_params - am_params) + ['idx']
            msg = f'Following <{name}> parameters in addfile are overwriten: '
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

        self.link = make_link_table(self.adsys)
        logger.warning(f'AMS system {hex(id(self.amsys))} is linked to the ANDES system {hex(id(adsys))}.')

    @property
    def is_tds(self):
        """
        Indicator of whether the ANDES system is running a TDS.
        This property will return ``True`` as long as TDS is initialized.

        Check ``adsys.tds.TDS.init()`` for more details.
        """
        return bool(self.adsys.TDS.initialized)

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

    def _sync_check(self, amsys, adsys):
        """
        Check if AMS and ANDES systems are ready for sync.
        """
        if amsys.dyn.adsys:
            if amsys.dyn.adsys != adsys:
                logger.error('Target ANDES system is different from the linked one, quit.')
                return False
        if not amsys.is_setup:
            amsys.setup()
        if not adsys.is_setup:
            adsys.setup()
        if amsys.dyn.link is None:
            amsys.dyn.link = make_link_table(adsys=adsys)

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
        self._sync_check(amsys=sp, adsys=sa)

        # --- Information ---
        rtn = sp.recent if routine is None else getattr(sp, routine)
        if rtn is None:
            logger.warning('No assigned or recent solved routine found, quit send.')
            return False
        elif rtn.exit_code != 0:
            logger.warning(f'{sp.recent.class_name} is not solved at optimal, quit send.')
            return False
        else:
            logger.info(f'Send <{rtn.class_name}> results to ANDES <{hex(id(sa))}>...')

        # NOTE: if DC type, check if results are converted
        if (rtn.type != 'ACED') and (not rtn.is_ac):
            logger.error(f'<{rtn.class_name}> AC conversion failed or not done yet!')

        # --- Mapping ---
        map2 = getattr(rtn, 'map2')     # mapping-to dict
        if len(map2) == 0:
            logger.warning(f'{rtn.class_name} has empty map2, quit send.')
            return True

        # NOTE: ads is short for ANDES
        for vname_ams, (mname_ads, pname_ads) in map2.items():
            mdl_ads = getattr(sa, mname_ads)  # ANDES model or group

            # --- skipping scenarios ---
            if mdl_ads.n == 0:
                logger.debug(f'ANDES model <{mname_ads}> is empty.')
                continue

            var_ams = getattr(rtn, vname_ams)  # instance of AMS routine var
            idx_ads = var_ams.get_idx()  # use AMS idx as target ANDES idx

            # --- special scenarios ---
            # 1. gen online status; in TDS running, setting u is invalid
            cond_ads_stg_u = (mname_ads in ['StaticGen', 'PV', 'Sclak']) and (pname_ads == 'u')
            if cond_ads_stg_u and (self.is_tds):
                logger.info(f'Skip sending {vname_ams} to StaticGen.u during TDS')
                continue

            # 2. Bus voltage
            cond_ads_bus_v0 = (mname_ads == 'Bus') and (pname_ads == 'v0')
            if cond_ads_bus_v0 and (self.is_tds):
                logger.info(f'Skip sending {vname_ams} t0 Bus.v0 during TDS')
                continue

            # 3. gen power reference; in TDS running, pg should go to TurbineGov
            cond_ads_stg_p0 = (mname_ads in ['StaticGen', 'PV', 'Sclak']) and (pname_ads == 'p0')
            if cond_ads_stg_p0 and (self.is_tds):
                # --- SynGen: TurbineGov.pref0 ---
                syg_idx = sp.dyn.link['syg_idx'].dropna().tolist()  # SynGen idx
                # corresponding StaticGen idx in ANDES
                stg_syg_idx = sa.SynGen.get(src='gen', attr='v', idx=syg_idx,
                                            allow_none=True, default=None)
                # corresponding TurbineGov idx in ANDES
                gov_idx = sa.TurbineGov.find_idx(keys='syn', values=syg_idx)
                # corresponding StaticGen pg in AMS
                syg_ams = rtn.get(src='pg', attr='v', idx=stg_syg_idx)
                # NOTE: check consistency
                syg_mask = self.link['syg_idx'].notnull() & self.link['gov_idx'].isnull()
                if syg_mask.any():
                    logger.debug('Governor is not complete for SynGen.')
                # --- pref ---
                sa.TurbineGov.set(value=syg_ams, idx=gov_idx, src='pref0', attr='v')

                # --- DG: DG.pref0 ---
                dg_idx = sp.dyn.link['dg_idx'].dropna().tolist()  # DG idx
                # corresponding StaticGen idx in ANDES
                stg_dg_idx = sa.DG.get(src='gen', attr='v', idx=dg_idx,
                                       allow_none=True, default=None)
                # corresponding StaticGen pg in AMS
                dg_ams = rtn.get(src='pg', attr='v', idx=stg_dg_idx)
                # --- pref ---
                sa.DG.set(value=dg_ams, idx=dg_idx, src='pref0', attr='v')

                # --- RenGen: seems unnecessary ---
                # TODO: which models/params are used to control output and auxillary power?

                var_dest = ''
                if len(syg_ams) > 0:
                    var_dest = 'TurbineGov.pref0'
                if len(dg_ams) > 0:
                    var_dest += ' and DG.pref0'
                logger.warning(f'Send <{vname_ams}> to {var_dest}')
                continue

            # --- other scenarios ---
            if _dest_check(mname=mname_ads, pname=pname_ads, idx=idx_ads, adsys=sa):
                mdl_ads.set(src=pname_ads, attr='v', idx=idx_ads, value=var_ams.v)
                logger.warning(f'Send <{vname_ams}> to {mname_ads}.{pname_ads}')
        return True

    def receive(self, adsys=None, routine=None, no_update=False):
        """
        Receive ANDES system results to AMS devices.

        Parameters
        ----------
        adsys : adsys.System.system, optional
            The target ANDES dynamic system instance. If not provided, use the
            linked ANDES system isntance (``sp.dyn.adsys``).
        routine : str, optional
            The routine to be received from ANDES. If None, ``recent`` will be used.
        no_update : bool, optional
            True to skip update the AMS routine parameters after sync. Default is False.
        """
        sa = adsys if adsys is not None else self.adsys
        sp = self.amsys
        self._sync_check(amsys=sp, adsys=sa)

        #  --- Information: not necessary in receive ---
        rtn = sp.recent if routine is None else getattr(sp, routine)
        if rtn is None:
            logger.warning('No assigned or recent solved routine found, quit receive.')
            return False

        # --- Mapping ---
        map1 = getattr(rtn, 'map1')     # mapping-from dict
        if len(map1) == 0:
            logger.warning(f'{rtn.class_name} has empty map1, quit receive.')
            return True

        link = sp.dyn.link      # link table
        pname_to_update = []    # list of parameters to be updated
        for vname_ams, (mname_ads, pname_ads) in map1.items():
            mdl_ads = getattr(sa, mname_ads)  # ANDES model or group

            # --- skipping scenarios ---
            if mdl_ads.n == 0:
                logger.debug(f'ANDES model <{mname_ads}> is empty.')
                continue

            idx_ads = rtn.__dict__[vname_ams].get_idx()  # use AMS idx as target ANDES idx

            # --- special scenarios ---
            # 1. gen online status; in TDS running, take from dynamic generator
            cond_ads_stg_u = (mname_ads in ['StaticGen', 'PV', 'Sclak']) and (pname_ads == 'u')
            if cond_ads_stg_u and (self.is_tds):
                # --- SynGen ---
                u_sg = sa.SynGen.get(idx=link['syg_idx'].replace(np.NaN, None).to_list(),
                                     src='u', attr='v',
                                     allow_none=True, default=0,)
                # --- DG ---
                u_dg = sa.DG.get(idx=link['dg_idx'].replace(np.NaN, None).to_list(),
                                 src='u', attr='v',
                                 allow_none=True, default=0,)
                # --- RenGen ---
                u_rg = sa.RenGen.get(idx=link['rg_idx'].replace(np.NaN, None).to_list(),
                                     src='u', attr='v',
                                     allow_none=True, default=0,)
                # --- output ---
                u_dyg = u_sg + u_rg + u_dg
                # NOTE: StaticGen that has no dynamic generator
                # Sync StaticGen.u first, then overwrite the ones with dynamic generator
                u_stg = sa.StaticGen.get(src='u', attr='v',
                                         idx=link['stg_idx'].values)
                # NOTE: only update u if changed actually
                u0_rtn = rtn.get(src=vname_ams, attr='v', idx=link['stg_idx'].values).copy()
                rtn.set(src=vname_ams, attr='v', value=u_stg,
                        idx=link['stg_idx'].values)
                rtn.set(src=vname_ams, attr='v', value=u_dyg,
                        idx=link['stg_idx'].values)
                u_rtn = rtn.get(src=vname_ams, attr='v', idx=link['stg_idx'].values).copy()
                if not np.array_equal(u0_rtn, u_rtn):
                    pname_to_update.append(vname_ams)
                var_dest = ''
                var_dest += 'SynGen.u' if sa.SynGen.n > 0 else ''
                var_dest += ', DG.u' if sa.DG.n > 0 else ''
                var_dest += ', RenGen.u' if sa.RenGen.n > 0 else ''
                var_dest += ', StaicGen.u' if sa.RenGen.n+sa.SynGen.n+sa.DG.n < sa.StaticGen.n else ''
                logger.info(f'Receive <{vname_ams}> from {var_dest}')
                continue

            # 2. gen output power; in TDS running, take from dynamic generator
            cond_ads_stg_p = (mname_ads in ['StaticGen', 'PV', 'Sclak']) and (pname_ads == 'p')
            if cond_ads_stg_p and (self.is_tds):
                # --- SynGen ---
                p_sg = sa.SynGen.get(idx=link['syg_idx'].replace(np.NaN, None).to_list(),
                                     src='Pe', attr='v',
                                     allow_none=True, default=0,)
                # --- DG ---
                Ie_dg = sa.DG.get(idx=link['dg_idx'].replace(np.NaN, None).to_list(),
                                  src='Ipout_y', attr='v',
                                  allow_none=True, default=0,)
                v_dg = sa.DG.get(idx=link['dg_idx'].replace(np.NaN, None).to_list(),
                                 src='v', attr='v',
                                 allow_none=True, default=0,)
                p_dg = Ie_dg * v_dg
                # --- RenGen ---
                p_rg = sa.RenGen.get(idx=link['rg_idx'].replace(np.NaN, None).to_list(),
                                     src='Pe', attr='v',
                                     allow_none=True, default=0,)
                # --- output ---
                p_dyg = p_sg + p_rg + p_dg
                # NOTE: StaticGen that has no dynamic generator
                # Sync StaticGen.p first, then overwrite the ones with dynamic generator
                p_stg = sa.StaticGen.get(src='p', attr='v',
                                         idx=link['stg_idx'].values)
                rtn.set(src=vname_ams, attr='v', value=p_stg,
                        idx=link['stg_idx'].values)
                rtn.set(src=vname_ams, attr='v', value=p_dyg,
                        idx=link['stg_idx'].values)

                pname_to_update.append(vname_ams)

                var_dest = ''
                var_dest += 'SynGen.Pe' if sa.SynGen.n > 0 else ''
                var_dest += ', DG.Pe' if sa.DG.n > 0 else ''
                var_dest += ', RenGen.Pe' if sa.RenGen.n > 0 else ''
                var_dest += ', StaicGen.p' if sa.RenGen.n+sa.SynGen.n+sa.DG.n < sa.StaticGen.n else ''
                logger.info(f'Receive <{vname_ams}> from {var_dest}')
                continue

            # --- other scenarios ---
            if _dest_check(mname=mname_ads, pname=pname_ads, idx=idx_ads, adsys=sa):
                v_ads = mdl_ads.get(src=pname_ads, attr='v', idx=idx_ads)
                rtn.set(src=vname_ams, attr='v', idx=idx_ads, value=v_ads)
                pname_to_update.append(vname_ams)
                logger.warning(f'Receive <{vname_ams}> from {mname_ads}.{pname_ads}')

        # --- update routine parameters ---
        if no_update and (len(pname_to_update) > 0):
            logger.info(f'Please update <{rtn.class_name}> parameters: {pname_to_update}')
        elif len(pname_to_update) > 0:
            rtn.update(params=pname_to_update, mat_make=False)
        return True


def _dest_check(mname, pname, idx, adsys):
    """
    Check if destination is valid.

    Parameters
    ----------
    mname : str
        Target ANDES model/group name.
    pname : str
        Target ANDES parameter name.
    idx : list
        Target idx.
    adsys : ANDES.system.System
        Target ANDES system.
    """
    # --- check model ---
    if not hasattr(adsys, mname):
        raise ValueError(f'Model error: ANDES system <{hex(adsys)}> has no <{mname}>')

    # --- check param ---
    mdl = getattr(adsys, mname)
    _is_grp = mname in adsys.groups.keys()
    # if it is a group, iterate through all models in the group
    mdls_grp_name = list(adsys.groups[mname].models.keys()) if _is_grp else [mname]
    n_no_mdl = 0
    for mdl_name in mdls_grp_name:
        mdl_to_check = getattr(adsys, mdl_name, None)
        if mdl_to_check is not None and hasattr(mdl_to_check, pname):
            break   # found the param in any one of the models, break
        n_no_mdl += 1
    if n_no_mdl == len(mdls_grp_name):
        raise ValueError(f'Param error: ANDES <{mdl.class_name}> has no <{pname}>')

    # --- check idx ---
    if _is_grp:
        _ads_idx = [v for mdl in mdl.models.values() for v in mdl.idx.v]
    else:
        _ads_idx = mdl.idx.v
    if not set(idx).issubset(set(_ads_idx)):
        idx_gap = set(idx) - set(_ads_idx)
        raise ValueError(f'Idx error: ANDES <{mdl.class_name}> has no idx: {idx_gap}')

    return True


def build_group_table(adsys, grp_name, param_name, mdl_name=None):
    """
    Build the table for devices in a group in an ANDES System.

    Parameters
    ----------
    adsys : andes.system.System
        The ANDES system to build the table
    grp_name : string
        The ANDES group
    param_name : list of string
        The common columns of a group that to be included in the table.
    mdl_name : list of string
        The list of models that to be included in the table. Default as all models.

    Returns
    -------
    DataFrame

        The output Dataframe contains the columns from the device
    """
    grp_df = pd.DataFrame(columns=param_name)
    grp = getattr(adsys, grp_name)  # get the group instance

    mdl_to_add = mdl_name if mdl_name else list(grp.models.keys())
    mdl_dfs = [getattr(adsys, mdl).as_df()[param_name] for mdl in mdl_to_add]

    grp_df = pd.concat(mdl_dfs, axis=0, ignore_index=True)

    # --- type sanity check ---
    mdl_1st = adsys.models[mdl_to_add[0]]
    # NOTE: force IdxParam to be string type
    cols_to_convert = [col for col in param_name if
                       (mdl_1st.params[col].class_name == 'IdxParam') and
                       (pd.api.types.is_numeric_dtype(grp_df[col]))]
    # NOTE: if 'idx' is included, force it to be string type
    if ('idx' in param_name) and pd.api.types.is_numeric_dtype(grp_df['idx']):
        cols_to_convert.append('idx')

    grp_df[cols_to_convert] = grp_df[cols_to_convert].astype(int).astype(str)
    return grp_df


def make_link_table(adsys):
    """
    Build the link table for generators and generator controllers in an ANDES
    System, including ``SynGen`` and ``DG`` for now.

    Parameters
    ----------
    adsys : andes.system.System
        The ANDES system to link

    Returns
    -------
    DataFrame

        Each column in the output Dataframe contains the ``idx`` of linked
        ``StaticGen``, ``Bus``, ``DG``, ``RenGen``, ``RenExciter``, ``SynGen``,
        ``Exciter``, and ``TurbineGov``, ``gammap``, ``gammaq``.
    """
    # --- build group tables ---
    # 1) StaticGen
    ssa_stg = build_group_table(adsys=adsys, grp_name='StaticGen',
                                param_name=['u', 'name', 'idx', 'bus'],
                                mdl_name=[])
    # 2) TurbineGov
    ssa_gov = build_group_table(adsys=adsys, grp_name='TurbineGov',
                                param_name=['idx', 'syn'],
                                mdl_name=[])
    # 3) Exciter
    ssa_exc = build_group_table(adsys=adsys, grp_name='Exciter',
                                param_name=['idx', 'syn'],
                                mdl_name=[])
    # 4) SynGen
    ssa_syg = build_group_table(adsys=adsys, grp_name='SynGen', mdl_name=['GENCLS', 'GENROU'],
                                param_name=['idx', 'bus', 'gen', 'gammap', 'gammaq'])
    # 5) DG
    ssa_dg = build_group_table(adsys=adsys, grp_name='DG', mdl_name=[],
                               param_name=['idx', 'bus', 'gen', 'gammap', 'gammaq'])
    # 6) RenGen
    ssa_rg = build_group_table(adsys=adsys, grp_name='RenGen', mdl_name=[],
                               param_name=['idx', 'bus', 'gen', 'gammap', 'gammaq'])
    # 7) RenExciter
    ssa_rexc = build_group_table(adsys=adsys, grp_name='RenExciter', mdl_name=[],
                                 param_name=['idx', 'reg'])

    # --- build link table ---
    # NOTE: use bus index as unique identifier
    ssa_bus = build_group_table(adsys=adsys, grp_name='ACTopology', mdl_name=['Bus'],
                                param_name=['name', 'idx'])
    # 1) StaticGen
    ssa_key = pd.merge(left=ssa_stg.rename(columns={'name': 'stg_name', 'idx': 'stg_idx',
                                                    'bus': 'bus_idx', 'u': 'stg_u'}),
                       right=ssa_bus.rename(columns={'name': 'bus_name', 'idx': 'bus_idx'}),
                       how='left', on='bus_idx')
    # 2) Dynamic Generators
    ssa_syg = pd.merge(left=ssa_key, how='right', on='stg_idx',
                       right=ssa_syg.rename(columns={'idx': 'syg_idx', 'gen': 'stg_idx'}))
    ssa_dg = pd.merge(left=ssa_key, how='right', on='stg_idx',
                      right=ssa_dg.rename(columns={'idx': 'dg_idx', 'gen': 'stg_idx'}))
    ssa_rg = pd.merge(left=ssa_key, how='right', on='stg_idx',
                      right=ssa_rg.rename(columns={'idx': 'rg_idx', 'gen': 'stg_idx'}))

    # NOTE: for StaticGen without Dynamic Generator, fill gammap and gammaq as 1
    ssa_key0 = pd.merge(left=ssa_key, how='left', on='stg_idx',
                        right=ssa_syg[['stg_idx', 'syg_idx']])
    ssa_key0 = pd.merge(left=ssa_key0, how='left', on='stg_idx',
                        right=ssa_dg[['stg_idx', 'dg_idx']])
    ssa_key0 = pd.merge(left=ssa_key0, how='left', on='stg_idx',
                        right=ssa_rg[['stg_idx', 'rg_idx']])

    ssa_key0 = ssa_key0.fillna(value=False)
    dyr = ssa_key0['syg_idx'].astype(bool) + ssa_key0['dg_idx'].astype(bool) + ssa_key0['rg_idx'].astype(bool)
    non_dyr = np.logical_not(dyr)
    ssa_dyr0 = ssa_key0[non_dyr]
    ssa_dyr0['gammap'] = 1
    ssa_dyr0['gammaq'] = 1

    ssa_key = pd.concat([ssa_syg, ssa_dg, ssa_rg, ssa_dyr0], axis=0)
    ssa_key = pd.merge(left=ssa_key,
                       right=ssa_exc.rename(columns={'idx': 'exc_idx', 'syn': 'syg_idx'}),
                       how='left', on='syg_idx')
    ssa_key = pd.merge(left=ssa_key,
                       right=ssa_gov.rename(columns={'idx': 'gov_idx', 'syn': 'syg_idx'}),
                       how='left', on='syg_idx')
    ssa_key = pd.merge(left=ssa_key, how='left', on='rg_idx',
                       right=ssa_rexc.rename(columns={'idx': 'rexc_idx', 'reg': 'rg_idx'}))

    # NOTE: other cols might be useful in the future
    # cols = ['stg_name', 'stg_u', 'stg_idx', 'bus_idx', 'dg_idx', 'rg_idx', 'rexc_idx',
    #         'syg_idx', 'exc_idx', 'gov_idx', 'bus_name', 'gammap', 'gammaq']
    # re-order columns
    cols = ['stg_idx', 'bus_idx',               # static gen
            'syg_idx', 'gov_idx',               # syn gen
            'dg_idx',                           # distributed gen
            'rg_idx',                           # renewable gen
            'gammap', 'gammaq',                 # gamma
            ]
    out = ssa_key[cols].sort_values(by='stg_idx', ascending=False).reset_index(drop=True)
    return out
