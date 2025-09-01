"""
Json reader and writer for AMS.

This module leverages the existing parser and writer in andes.io.json.
"""
import json
import logging

from collections import OrderedDict

from andes.io.json import (testlines, read)  # NOQA
from andes.utils.paths import confirm_overwrite

from ams.shared import pd
from ams.shared import summary_row, summary_name, model2df


logger = logging.getLogger(__name__)


def write(system, outfile, skip_empty=True, overwrite=None,
          to_andes=False,
          ):
    """
    Write loaded AMS system data into an json file.
    If to_andes is True, only write models that are in ANDES,
    but the outfile might not be able to be read back into AMS.

    Revised from `andes.io.json.write`, where non-andes models
    are skipped if `to_andes` is True.

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
    out = OrderedDict()
    summary_found = False
    for name, instance in system.models.items():
        df = model2df(instance, skip_empty, to_andes=to_andes)
        if df is None:
            continue

        if name == summary_name:
            df = pd.concat([pd.DataFrame([summary_row]), df], ignore_index=True)
            df.index.name = "uid"
            summary_found = True
        out[name] = df.to_dict(orient=orient)

    if not summary_found:
        df = pd.DataFrame([summary_row])
        df.index.name = "uid"
        out[summary_name] = df.to_dict(orient=orient)

    return json.dumps(out, indent=2)
