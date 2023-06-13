"""
Interface with ANDES
"""

import os
import logging

from andes.shared import pd, np
from andes.utils.misc import elapsed
from andes import load as andes_load

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
        syngen_idx = sa.SynGen.find_idx(keys='bus', values=sa.Bus.idx.v, allow_none=True, default=None)
        syngen_idx = [x for x in syngen_idx if x is not None]

        stg_idx = sa.StaticGen.find_idx(keys='bus', values=sa.Bus.idx.v, allow_none=True, default=None)
        stg_idx = [x for x in stg_idx if x is not None]

        # SynGen gen with StaticGen idx
        if any(item not in stg_idx for item in syngen_idx):
            logger.debug("Correct SynGen idx to match StaticGen.")
            sa.SynGen.set(src='gen', idx=syngen_idx, attr='v', value=stg_idx)

        # SynGen Vn with Bus Vn
        syngen_bus = sa.SynGen.get(src='bus', idx=syngen_idx, attr='v')

        gen_vn = sa.SynGen.get(src='Vn', idx=syngen_idx, attr='v')
        bus_vn = sa.Bus.get(src='Vn', idx=syngen_bus, attr='v')
        if not np.equal(gen_vn, bus_vn).all():
            sa.SynGen.set(src='Vn', idx=syngen_idx, attr='v', value=bus_vn)
            logger.warning('Correct SynGen Vn to match Bus Vn.')

        # FIXME: add consistency check for RenGen, to SynGen

        _, s = elapsed(t)
        logger.info('Addfile parsed in %s.', s)

    if setup:
        sa.setup()

    system.dyn = Dynamic(sp=system, sa=sa)
    logger.info(f'AMS system {hex(id(system))} link to ANDES system {hex(id(sa))}')
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
        pass

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
