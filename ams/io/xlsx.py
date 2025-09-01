"""
Excel reader and writer for AMS.

This module leverages the existing parser and writer in andes.io.xlsx.
"""
import logging

from andes.io.xlsx import (read, testlines, confirm_overwrite, _add_book)  # NOQA

from ams.shared import pd
from ams.shared import summary_row, summary_name, model2df


logger = logging.getLogger(__name__)


def write(system, outfile,
          skip_empty=True, overwrite=None, add_book=None,
          to_andes=False,
          ):
    """
    Write loaded AMS system data into an xlsx file

    Revised from `andes.io.xlsx.write`, where non-ANDES models
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

    Revised from `andes.io.xlsx._write_system`, where non-ANDES models
    are skipped if `to_andes` is True.
    """
    summary_found = False
    for name, instance in system.models.items():
        df = model2df(instance, skip_empty, to_andes=to_andes)
        if df is None:
            continue

        if name == summary_name:
            df = pd.concat([pd.DataFrame([summary_row]), df], ignore_index=True)
            df.index.name = "uid"
            summary_found = True

        df.to_excel(writer, sheet_name=name, freeze_panes=(1, 0))

    if not summary_found:
        df = pd.DataFrame([summary_row])
        df.index.name = "uid"
        df.to_excel(writer, sheet_name=summary_name, freeze_panes=(1, 0))

    return writer
