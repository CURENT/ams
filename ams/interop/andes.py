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

logger = logging.getLogger(__name__)


def to_andes(system,
             setup=True,
             addfile=None,
             overwrite=None,
             keep=False,
             **kwargs):
    """
    Convert the current model to ANDES format.

    Parameters
    ----------
    system: System
        The AMS system to be converted to ANDES format.
    setup: bool
        Whether to call `setup()` after the conversion.
    addfile: str
        The additional file to be converted to ANDES format.
    overwrite: bool
        Whether to overwrite the existing file.
    keep: bool
        Whether to keep the converted file.
    kwargs: dict
        Keyword arguments to be passed to `andes.system.System`.
    """
    t0, _ = elapsed()
    andes_file = system.files.name + '.json'

    json.write(system, andes_file,
               overwrite=overwrite,
               to_andes=True,
               )

    _, s = elapsed(t0)
    logger.info(f'System convert to ANDES in {s}, save to "{andes_file}".')

    sa = andes_load(andes_file, setup=False, **kwargs)

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
                    sa.add(name, row)

            # --- for debugging ---
            sa.df_in = df_models

        else:
            logger.warning('Addfile format "%s" is not supported yet.', add_format)
            # FIXME: xlsx input file with dyr addfile result into KeyError: 'Toggle'
            # add_parser = importlib.import_module('andes.io.' + add_format)
            # if not add_parser.read_add(system, addfile):
            #     logger.error('Error parsing addfile "%s" with %s parser.', addfile, add_format)

        # --- consistency check ---
        stg_idx = sa.StaticGen.find_idx(keys='bus', values=sa.Bus.idx.v, allow_none=True, default=None)
        stg_idx = [x for x in stg_idx if x is not None]

        syg_idx = sa.SynGen.find_idx(keys='bus', values=sa.Bus.idx.v, allow_none=True, default=None)
        syg_idx = [x for x in syg_idx if x is not None]

        rg_idx = sa.RenGen.find_idx(keys='bus', values=sa.Bus.idx.v, allow_none=True, default=None)
        rg_idx = [x for x in rg_idx if x is not None]

        dg_idx = sa.DG.find_idx(keys='bus', values=sa.Bus.idx.v, allow_none=True, default=None)
        dg_idx = [x for x in dg_idx if x is not None]

        syg_bus = sa.SynGen.get(src='bus', idx=syg_idx, attr='v')
        rg_bus = sa.RenGen.get(src='bus', idx=rg_idx, attr='v')
        dg_bus = sa.DG.get(src='bus', idx=dg_idx, attr='v')

        # SynGen gen with StaticGen idx
        if any(item not in stg_idx for item in syg_idx):
            logger.debug("Correct SynGen idx to match StaticGen.")
            syg_stg = sa.StaticGen.find_idx(keys='bus', values=syg_bus, allow_none=True, default=None)
            sa.SynGen.set(src='gen', idx=syg_idx, attr='v', value=syg_stg)
        if any(item not in stg_idx for item in rg_idx):
            logger.debug("Correct RenGen idx to match StaticGen.")
            rg_stg = sa.StaticGen.find_idx(keys='bus', values=rg_bus, allow_none=True, default=None)
            sa.RenGen.set(src='gen', idx=rg_idx, attr='v', value=rg_stg)
        if any(item not in stg_idx for item in dg_idx):
            logger.debug("Correct DG idx to match StaticGen.")
            dg_stg = sa.StaticGen.find_idx(keys='bus', values=dg_bus, allow_none=True, default=None)
            sa.DG.set(src='gen', idx=dg_idx, attr='v', value=dg_stg)

        # SynGen Vn with Bus Vn
        syg_vn = sa.SynGen.get(src='Vn', idx=syg_idx, attr='v')
        bus_vn = sa.Bus.get(src='Vn', idx=syg_bus, attr='v')
        if not np.equal(syg_vn, bus_vn).all():
            sa.SynGen.set(src='Vn', idx=syg_idx, attr='v', value=bus_vn)
            logger.warning('Correct SynGen Vn to match Bus Vn.')

        # NOTE: RenGen and DG do not have Vn

        _, s = elapsed(t)
        logger.info('Addfile parsed in %s.', s)

    if setup:
        sa.setup()

    t0, _ = elapsed()
    system.dyn = Dynamic(sp=system, sa=sa)
    _, s = elapsed(t0)
    logger.info(f'AMS system {hex(id(system))} link to ANDES system {hex(id(sa))} in %s.', s)
    return sa


class Dynamic:
    """
    The ANDES center class.
    """

    def __init__(self,
                 sp=None,
                 sa=None,
                 ) -> None:
        self.sp = sp  # AMS system
        self.sa = sa  # ANDES system

        # TODO: add summary table
        self.link = None  # ANDES system link table
        self.make_link()
        pass

    @property
    def is_tds(self):
        """
        Check if ANDES system has run a TDS.
        """
        return self.sa.TDS.initialized & bool(self.sa.dae.t)

    def make_link(self):
        """
        Make the link table of the ANDES system.

        A wrapper of `andes.interop.pandapower.make_link_table`.
        """
        self.link = make_link_table(self.sa)
        # adjust columns
        cols = ['stg_idx', 'bus_idx',               # static gen
                'syg_idx', 'gov_idx',               # syn gen
                'dg_idx',                           # distributed gen
                'rg_idx',                           # renewable gen
                'gammap', 'gammaq',                 # gamma
                ]
        self.link = self.link[cols]

    def send_dgu(self):
        """
        Send to ANDES the dynamic generator online status.

        # TODO: add support for switch dynamic gen online status
        """
        sa = self.sa
        sp = self.sp
        return True

    def send_tgr(self):
        """
        Sned to TurbinGov refrence.
        """
        sa = self.sa
        sp = self.sp
        # 1) TurbineGov
        syg_idx = sp.dyn.link['syg_idx'].dropna().tolist()  # SynGen idx
        # corresponding StaticGen idx in ANDES
        stg_syg_idx = sa.SynGen.get(src='gen', attr='v', idx=syg_idx,
                                    allow_none=True, default=None,
                                    )
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
        # TODO: paux, using Pext0

        # 3) RenGen
        # TODO: seems to be unnecessary

        return True

    def send(self):
        """
        Send the AMS results to ANDES.
        """
        # --- send routine results ---
        try:
            rtn_name = self.sp.recent.class_name
            logger.debug(f'Sending {rtn_name} results to ANDES...')
        except AttributeError:
            logger.warning('No solved AMS routine found. Unable to sync with ANDES.')
            return False
        # FIXME: dynamic device online status?
        # --- send models online status ---
        if self.is_tds:     # sync dynamic device
            logger.warning('Dynamic has been initialized, Bus angle and voltage are skipped.')
        for mname, mdl in self.sp.models.items():
            if mdl.n == 0:
                continue
            if mdl.group == 'StaticGen':
                if self.is_tds:     # sync dynamic device
                    self.send_dgu()
                    self.send_tgr()
                    # once dynamic has started, skip bus
                    if mdl.class_name == 'Bus':
                        continue
                    # TODO: add other dynamic info
                    continue
            idx = mdl.idx.v
            if mname in self.sa.models:
                mdl_andes = getattr(self.sa, mname)
                u_ams = mdl.get(src='u', idx=idx, attr='v')
                mdl_andes.set(src='u', idx=idx, attr='v', value=u_ams)

        map2 = getattr(self.sp.recent, 'map2')
        if len(map2) == 0:
            logger.warning(f'Mapping dict "map2" of {self.sp.recent.class_name} is empty.')
            return True

        # FIXME: not consider dynamic device yet
        for mname, pmap in map2.items():
            mdl = getattr(self.sa, mname)
            for ams_var, andes_param in pmap.items():
                idx_ams = getattr(self.sp.recent, ams_var).get_idx()
                v_ams = getattr(self.sp.recent, ams_var).v
                try:
                    mdl.set(src=andes_param, attr='v',
                            idx=idx_ams, value=v_ams,
                            )
                except KeyError:
                    logger.warning(f'ANDES: Param {andes_param} not found in Model {mname}.')
                    continue
        return True

    def receive(self):
        """
        Receive the AMS system from the ANDES system.
        """
        # --- receive device online status ---
        try:
            rtn_name = self.sp.recent.class_name
            logger.debug(f'Receiving ANDES results to {rtn_name}.')
        except AttributeError:
            logger.warning('No target AMS routine found. Unable to sync with ANDES.')
            return False
        for mname, mdl in self.sp.models.items():
            if mdl.n == 0:
                continue
            if mdl.group == 'StaticGen':
                if self.is_tds:     # sync dynamic device
                    # --- dynamic generator online status ---
                    self.receive_dgu()
                    self.receive_pe()
                    # TODO: add other dynamic info
                    continue
            idx = mdl.idx.v
            if mname in self.sa.models:
                mdl_andes = getattr(self.sa, mname)
                u_andes = mdl_andes.get(src='u', idx=idx, attr='v')
                mdl.set(src='u', idx=idx, attr='v', value=u_andes)

        # TODO: dynamic results
        map1 = getattr(self.sp.recent, 'map1')
        if len(map1) == 0:
            logger.warning(f'Mapping dict "map1" of {self.sp.recent.class_name} is empty.')
            return True
        return True

    def receive_pe(self):
        """
        Get the dynamic generator output power.
        """
        if not self.is_tds:     # sync dynamic device
            logger.warning('Dynamic is not running, receiving Pe is skipped.')
            return True
        # TODO: get pe from ANDES
        # 1) SynGen
        # using Pe

        # 2) DG
        # using Pe

        # 3) RenGen
        # using v and Ipout_y
        # sa.PVD1.get(src='Ipout_y', attr='v', idx=[])
        pass
        return True

    def receive_dgu(self):
        """
        Get the dynamic generator online status.
        """
        sa = self.sa
        sp = self.sp
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
