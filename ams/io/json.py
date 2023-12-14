"""
Json reader and writer for AMS.

This module leverages the existing parser and writer in andes.io.json.
"""
import json
import logging

from collections import OrderedDict

from andes.io.json import (testlines, read)  # NOQA
from andes.utils.paths import confirm_overwrite

from andes.system import System as andes_system

logger = logging.getLogger(__name__)


def write(system, outfile, skip_empty=True, overwrite=None,
          to_andes=False,
          ):
    """
    Write loaded AMS system data into an json file.
    If to_andes is True, only write models that are in ANDES,
    but the outfile might not be able to be read back into AMS.

    Revise function ``andes.io.json.write`` to skip non-andes models.

    Parameters
    ----------
    system : System
        A loaded system with parameters
    outfile : str
        Path to the output file
    skip_empty : bool
        Skip output of empty models (n = 0)
    overwrite : bool, optional
        None to prompt for overwrite selection; True to overwrite; False to not overwrite
    to_andes : bool, optional
        Write to an ANDES system, where non-ANDES models are skipped

    Returns
    -------
    bool
        True if file written; False otherwise
    """
    if not confirm_overwrite(outfile, overwrite):
        return False

    if hasattr(outfile, 'write'):
        outfile.write(_dump_system(system, skip_empty, to_andes=to_andes))
    else:
        with open(outfile, 'w') as writer:
            writer.write(_dump_system(system, skip_empty, to_andes=to_andes))
            logger.info('JSON file written to "%s"', outfile)

    return True


def _dump_system(system, skip_empty, orient='records', to_andes=False):
    """
    Dump parameters of each model into a json string and return
    them all in an OrderedDict.
    """
    ad_models = []
    if to_andes:
        # Instantiate an ANDES system
        sa = andes_system(setup=False, default_config=True,
                          codegen=False, autogen_stale=False,
                          no_undill=True,)
        ad_models = list(sa.models.keys())
    out = OrderedDict()
    for name, instance in system.models.items():
        if skip_empty and instance.n == 0:
            continue
        if to_andes:
            if name not in ad_models:
                continue
            # NOTE: ommit parameters that are not in ANDES
            skip_params = []
            ams_params = list(instance.params.keys())
            andes_params = list(sa.models[name].params.keys())
            skip_params = list(set(ams_params) - set(andes_params))
            df = instance.cache.df_in.drop(skip_params, axis=1, errors='ignore')
            out[name] = df.to_dict(orient=orient)
        else:
            df = instance.cache.df_in
            out[name] = df.to_dict(orient=orient)
    return json.dumps(out, indent=2)
