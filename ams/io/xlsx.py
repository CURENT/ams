"""
Excel reader and writer for AMS power system parameters.

Consistent with ANDES, spreadsheets are used for AMS input and output.
"""

import logging
import pandas as pd

def write(system, filename):
    """Write a system to an Excel spreadsheet"""
    pass
    return None

def read(system, infile):
    """
    Read a system from an Excel spreadsheet
    Read an xlsx file with AMS model data into an empty system

    Parameters
    ----------
    system : System
        Empty AMS system instance
    infile : str or file-like
        Path to the input file, or a file-like object

    Returns
    -------
    System:
        System instance after reading
    """
    df_models = pd.read_excel(infile,
                              sheet_name=None,
                              index_col=0,
                              engine='openpyxl',
                              )

    for name, df in df_models.items():
        # drop rows that all nan
        df.dropna(axis=0, how='all', inplace=True)
        for row in df.to_dict(orient='records'):
            continue  # DEBUG
            system.add(name, row)

    return None