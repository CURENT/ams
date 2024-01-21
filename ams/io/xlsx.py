"""
Excel reader and writer for AMS.

This module leverages the existing parser and writer in andes.io.xlsx.
"""
import logging

from andes.io.xlsx import (read, testlines, confirm_overwrite, _add_book)  # NOQA
from andes.shared import pd
from andes.system import System as andes_system


logger = logging.getLogger(__name__)


def write(system, outfile,
          skip_empty=True, overwrite=None, add_book=None,
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
    add_book : str, optional
        An optional model to be added to the output spreadsheet
    to_andes : bool, optional
        Write to an ANDES system, where non-ANDES models are skipped

    Returns
    -------
    bool
        True if file written; False otherwise
    """
    if not confirm_overwrite(outfile, overwrite=overwrite):
        return False

    writer = pd.ExcelWriter(outfile, engine='xlsxwriter')
    writer = _write_system(system, writer, skip_empty, to_andes=to_andes)
    writer = _add_book(system, writer, add_book)

    writer.close()

    logger.info('xlsx file written to "%s"', outfile)
    return True


def _write_system(system, writer, skip_empty, to_andes=False):
    """
    Write the system to pandas ExcelWriter

    Rewrite function ``andes.io.xlsx._write_system`` to skip non-andes sheets.
    """
    ad_models = []
    if to_andes:
        # Instantiate an ANDES system
        sa = andes_system(setup=False, default_config=True,
                          codegen=False, autogen_stale=False,
                          no_undill=True,)
        ad_models = list(sa.models.keys())
    for name, instance in system.models.items():
        if skip_empty and instance.n == 0:
            continue
        instance.cache.refresh("df_in")
        if to_andes:
            if name not in ad_models:
                continue
            # NOTE: ommit parameters that are not in ANDES
            skip_params = []
            ams_params = list(instance.params.keys())
            andes_params = list(sa.models[name].params.keys())
            skip_params = list(set(ams_params) - set(andes_params))
            df = instance.cache.df_in.drop(skip_params, axis=1, errors='ignore')
        else:
            df = instance.cache.df_in
        df.to_excel(writer, sheet_name=name, freeze_panes=(1, 0))
    return writer
