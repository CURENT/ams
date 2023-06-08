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
    Write loaded AMS system data into an xlsx file

    Revised function ``andes.io.xlsx.write`` to skip non-andes models.

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
            writer.write(_dump_system(system, skip_empty))
            logger.info('JSON file written to "%s"', outfile)

    return True


def _dump_system(system, skip_empty, orient='records', to_andes=True):
    """
    Dump parameters of each model into a json string and return
    them all in an OrderedDict.
    """

    na_models = []
    skip_params = []
    if to_andes:
        # Initialize an ANDES system
        sa = andes_system(setup=False, default_config=True,
                          codegen=False, autogen_stale=False,
                          no_undill=True,
                          )
    out = OrderedDict()
    for name, instance in system.models.items():
        if skip_empty and instance.n == 0:
            continue
        if to_andes:
            if name not in sa.models.keys():
                na_models.append(name)
                continue
        if to_andes:
            ams_params = list(instance.params.keys())
            andes_params = list(sa.models[name].params.keys())
            skip_params = list(set(ams_params) - set(andes_params))
        if skip_params:
            df = instance.cache.df_in.drop(skip_params, axis=1)
        else:
            df = instance.cache.df_in
        out[name] = df.to_dict(orient=orient)

    return json.dumps(out, indent=2)
