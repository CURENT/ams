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
                'syg_idx', 'gov_idx', 'exc_idx',    # syn gen
                'dg_idx',                           # distributed gen
                'rg_idx', 'rexc_idx',               # renewable gen
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
        Sned to ANDES Governor refrence.
        """
        sa = self.sa
        sp = self.sp
        pg_ams = sp.ACOPF.get(src='pg', attr='v',
                              idx=sp.dyn.link['stg_idx'].replace(np.NaN, None).to_list(),
                              allow_none=True, default=0)
        # --- check consistency ---
        mask = self.link['syg_idx'].notnull() & self.link['gov_idx'].isnull()
        if mask.any():
            logger.debug('Governor is missing for SynGen.')
            return False
        # --- set to TurbineGov pref ---
        sa.TurbineGov.set(value=pg_ams,
                          idx=sp.dyn.link['gov_idx'].replace(np.NaN, None).to_list(),
                          src='pref0', attr='v')
        # --- set to TurbineGov paux ---
        # TODO: sync paux

        return True

    def send(self):
        """
        Send the AMS results to ANDES.
        """
        # FIXME: dynamic device online status?
        # --- send models online status ---
        logger.debug(f'Sending model online status to ANDES...')
        for mname, mdl in self.sp.models.items():
            if mdl.n == 0:
                continue
            if mdl.group == 'StaticGen':
                if self.is_tds:     # sync dynamic device
                    self.send_dgu()
                    self.send_tgr()
                    # TODO: add other dynamic info
                    continue
            idx = mdl.idx.v
            if mname in self.sa.models:
                mdl_andes = getattr(self.sa, mname)
                u_ams = mdl.get(src='u', idx=idx, attr='v')
                mdl_andes.set(src='u', idx=idx, attr='v', value=u_ams)

        # --- send routine results ---
        try:
            rtn_name = self.sp.recent.class_name
            logger.debug(f'Sending {rtn_name} results to ANDES...')
        except AttributeError:
            logger.warning('No solved AMS routine found. Unable to sync with ANDES.')
            return False

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
        # FIXME: dynamic device online status?
        logger.debug(f'Receiving model online status from ANDES...')
        for mname, mdl in self.sp.models.items():
            if mdl.n == 0:
                continue
            if mdl.group == 'StaticGen':
                if self.is_tds:     # sync dynamic device
                    # --- dynamic generator online status ---
                    self.receive_dgu()
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

    def receive_dgu(self):
        """
        Get the dynamic generator online status.
        """
        sa = self.sa
        sp = self.sp
        # --- SynGen ---
        u_sg = sa.SynGen.get(idx=sp.dyn.link['syg_idx'].replace(np.NaN, None).to_list(),
                             src='u', attr='v',
                             allow_none=True, default=0,)
        # --- RenGen ---
        u_rg = sa.RenGen.get(idx=sp.dyn.link['rg_idx'].replace(np.NaN, None).to_list(),
                             src='u', attr='v',
                             allow_none=True, default=0,)
        # --- DG ---
        u_dg = sa.DG.get(idx=sp.dyn.link['dg_idx'].replace(np.NaN, None).to_list(),
                         src='u', attr='v',
                         allow_none=True, default=0,)
        # --- sync into AMS ---
        u_dyngen = u_sg + u_rg + u_dg
        sp.StaticGen.set(idx=sp.dyn.link['stg_idx'].replace(np.NaN, None).to_list(),
                         src='u', attr='v',
                         value=u_dyngen,)
        return True

    def sync(self):
        """
        Sync the AMS system with the ANDES system.
        """
        # --- from andes ---
        # TODO: determin what needs to be retrieved from ANDES

        # --- to andes ---
        try:
            rtn_name = self.sp.recent.class_name
            logger.debug(f'Sending {rtn_name} results into ANDES...')
        except AttributeError:
            logger.warning('No solved AMS routine found. Unable to sync with ANDES.')
            return False

        map1 = getattr(self.sp.recent, 'map1')

        for mdl_name, param_map in map1.items():
            mdl = getattr(self.sa, mdl_name)
            for ams_algeb, andes_param in param_map.items():
                # TODO: check idx consistency
                idx_ams = getattr(self.sp.recent, ams_algeb).get_idx()
                v_ams = getattr(self.sp.recent, ams_algeb).v
                mdl.set(src=andes_param, attr='v',
                        idx=idx_ams, value=v_ams,
                        )

        # TODO:
        # --- bus ---
        return True
