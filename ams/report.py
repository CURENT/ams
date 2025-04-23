"""
Module for report generation.
"""
import logging
from collections import OrderedDict
from typing import List, Dict, Optional

from andes.io.txt import dump_data
from andes.shared import np
from andes.utils.misc import elapsed

from ams import __version__ as version
from ams.shared import copyright_msg, nowarranty_msg, report_time

logger = logging.getLogger(__name__)


DECIMALS = 6


def report_info(system) -> list:
    info = list()
    info.append('AMS' + ' ' + version + '\n')
    info.append(f'{copyright_msg}\n\n')
    info.append(f"{nowarranty_msg}\n")
    info.append('Case file: ' + str(system.files.case) + '\n')
    info.append(f'Report time: {report_time}\n\n')
    return info


class Report:
    """
    Report class to store routine analysis reports.

    Notes
    -----
    Revised from the ANDES project (https://github.com/CURENT/andes).
    Original author: Hantao Cui
    License: GPL3
    """

    def __init__(self, system):
        self.system = system
        self.basic = OrderedDict()

    @property
    def info(self):
        return report_info(self.system)

    def update(self):
        """
        Update values based on the requested content
        """
        system = self.system
        self.basic.update({
            'Buses': system.Bus.n,
            'Generators': system.PV.n + system.Slack.n,
            'Loads': system.PQ.n,
            'Shunts': system.Shunt.n,
            'Lines': system.Line.n,
            'Transformers': np.count_nonzero(system.Line.trans.v == 1),
            'Areas': system.Area.n,
            'Zones': system.Zone.n,
        })

    def collect(self, rtn, horizon=None):
        """
        Collect report data.

        Parameters
        ----------
        rtn : Routine
            Routine object to collect data from.
        horizon : str, optional
            Timeslot to collect data from. Only single timeslot is supported.
        """
        text = list()
        header = list()
        row_name = list()
        data = list()

        if not rtn.converged:
            return text, header, row_name, data

        owners = collect_owners(rtn)
        owners = collect_vars(owners, rtn, horizon, DECIMALS)
        owners = collect_exprs(owners, rtn, horizon, DECIMALS)
        owners = collect_exprcs(owners, rtn, horizon, DECIMALS)

        dump_collected_data(owners, text, header, row_name, data)

        return text, header, row_name, data

    def write(self):
        """
        Write report to file.
        """
        system = self.system
        if system.files.no_output is True:
            return

        text = list()
        header = list()
        row_name = list()
        data = list()
        self.update()

        t, _ = elapsed()

        # --- system info section ---
        text.append(self.info)
        header.append(None)
        row_name.append(None)
        data.append(None)

        # --- system summary section ---
        text.append(['='*10 + ' System Statistics ' + '='*10 + '\n'])
        header.append(None)
        row_name.append(self.basic.keys())
        data.append(list(self.basic.values()))

        # --- rountine data section ---
        rtns_to_collect = [rtn for rtn in system.routines.values() if rtn.converged]
        for rtn in rtns_to_collect:
            # --- routine summary ---
            text.append(['='*30 + f' {rtn.class_name} ' + '='*30])
            header.append(None)
            row_name.append(None)
            data.append(None)
            if hasattr(rtn, 'timeslot'):
                for slot in rtn.timeslot.v:
                    # --- timeslot summary ---
                    text.append(['-'*28 + f' {slot} ' + '-'*28])
                    header.append(None)
                    row_name.append(None)
                    data.append(None)
                    text_sum, header_sum, row_name_sum, data_sum = self.collect(rtn, horizon=[slot])
                    # --- timeslot data ---
                    text.extend(text_sum)
                    header.extend(header_sum)
                    row_name.extend(row_name_sum)
                    data.extend(data_sum)
            else:
                # single-period
                text_sum, header_sum, row_name_sum, data_sum = self.collect(rtn)
                # --- routine extended ---
                text.append([''])
                row_name.append(
                    ['Generation', 'Load'])

                if hasattr(rtn, 'pd'):
                    pd = rtn.pd.v.sum().round(DECIMALS)
                else:
                    pd = rtn.system.PQ.p0.v.sum().round(DECIMALS)
                if hasattr(rtn, 'qd'):
                    qd = rtn.qd.v.sum().round(DECIMALS)
                else:
                    qd = rtn.system.PQ.q0.v.sum().round(DECIMALS)

                if not hasattr(rtn, 'qg'):
                    header.append(['P (p.u.)'])
                    Pcol = [rtn.pg.v.sum().round(DECIMALS), pd]
                    data.append([Pcol])
                else:
                    header.append(['P (p.u.)', 'Q (p.u.)'])
                    Pcol = [rtn.pg.v.sum().round(DECIMALS), pd]
                    Qcol = [rtn.qg.v.sum().round(DECIMALS), qd]
                    data.append([Pcol, Qcol])

                # --- routine data ---
                text.extend(text_sum)
                header.extend(header_sum)
                row_name.extend(row_name_sum)
                data.extend(data_sum)
        dump_data(text, header, row_name, data, system.files.txt)

        _, s = elapsed(t)
        logger.info(f'Report saved to "{system.files.txt}" in {s}.')


def dump_collected_data(owners: dict, text: List, header: List, row_name: List, data: List) -> None:
    """
    Dump collected data into the provided lists.

    Parameters
    ----------
    owners : dict
        Dictionary of owners.
    text : list
        List to append text data to.
    header : list
        List to append header data to.
    row_name : list
        List to append row names to.
    data : list
        List to append data to.
    """
    for key, val in owners.items():
        text.append([f'{key} DATA:\n'])
        row_name.append(val['idx'])
        header.append(val['header'])
        data.append(val['data'])


def collect_exprcs(owners: Dict, rtn, horizon: Optional[str], decimals: int) -> Dict:
    """
    Collect expression calculations and populate the data dictionary.

    Parameters
    ----------
    owners : dict
        Dictionary of owners.
    rtn : Routine
        Routine object to collect data from.
    horizon : str, optional
        Timeslot to collect data from. Only single timeslot is supported.
    decimals : int
        Number of decimal places to round the data.

    Returns
    -------
    dict
        Updated dictionary of owners with collected ExpressionCalc data.
    """
    for key, exprc in rtn.exprcs.items():
        if exprc.owner is None:
            continue
        owner_name = exprc.owner.class_name
        idx_v = owners[owner_name]['idx']
        header_v = key if exprc.unit is None else f'{key} ({exprc.unit})'
        try:
            data_v = rtn.get(src=key, attr='v', idx=idx_v, horizon=horizon).round(decimals)
        except Exception:
            data_v = [np.nan] * len(idx_v)
        owners[owner_name]['header'].append(header_v)
        owners[owner_name]['data'].append(data_v)

    return owners


def collect_exprs(owners: Dict, rtn, horizon: Optional[str], decimals: int) -> Dict:
    """
    Collect expressions and populate the data dictionary.

    Parameters
    ----------
    owners : dict
        Dictionary of owners.
    rtn : Routine
        Routine object to collect data from.
    horizon : str, optional
        Timeslot to collect data from. Only single timeslot is supported.
    decimals : int
        Number of decimal places to round the data.

    Returns
    -------
    dict
        Updated dictionary of owners with collected Expression data.
    """
    for key, expr in rtn.exprs.items():
        if expr.owner is None:
            continue
        owner_name = expr.owner.class_name
        idx_v = owners[owner_name]['idx']
        header_v = key if expr.unit is None else f'{key} ({expr.unit})'
        try:
            data_v = rtn.get(src=key, attr='v', idx=idx_v, horizon=horizon).round(decimals)
        except Exception:
            data_v = [np.nan] * len(idx_v)
        owners[owner_name]['header'].append(header_v)
        owners[owner_name]['data'].append(data_v)

    return owners


def collect_vars(owners: Dict, rtn, horizon: Optional[str], decimals: int) -> Dict:
    """
    Collect variables and populate the data dictionary.

    Parameters
    ----------
    owners : dict
        Dictionary of owners.
    rtn : Routine
        Routine object to collect data from.
    horizon : str, optional
        Timeslot to collect data from. Only single timeslot is supported.
    decimals : int
        Number of decimal places to round the data.

    Returns
    -------
    dict
        Updated dictionary of owners with collected Var data.
    """

    for key, var in rtn.vars.items():
        if var.owner is None:
            continue
        owner_name = var.owner.class_name
        idx_v = owners[owner_name]['idx']
        header_v = key if var.unit is None else f'{key} ({var.unit})'
        try:
            data_v = rtn.get(src=key, attr='v', idx=idx_v, horizon=horizon).round(decimals)
        except Exception:
            data_v = [np.nan] * len(idx_v)
        owners[owner_name]['header'].append(header_v)
        owners[owner_name]['data'].append(data_v)

    return owners


def collect_owners(rtn):
    """
    Initialize an owners dictionary for data collection.

    Returns
    -------
    dict
        A dictionary of initialized owners.
    """
    # initialize data section by model
    owners_all = ['Bus', 'Line', 'StaticGen',
                  'PV', 'Slack', 'RenGen',
                  'DG', 'ESD1', 'PVD1', 'VSG',
                  'StaticLoad']

    # Filter owners that exist in the system
    owners_e = list({
        var.owner.class_name for var in rtn.vars.values() if var.owner is not None
    }.union(
        expr.owner.class_name for expr in rtn.exprs.values() if expr.owner is not None
    ).union(
        exprc.owner.class_name for exprc in rtn.exprcs.values() if exprc.owner is not None
    ))

    # Use a dictionary comprehension to create vars_by_owner
    owners = {
        name: {'idx': [],
               'name': [],
               'header': [],
               'data': [], }
        for name in owners_all if name in owners_e and getattr(rtn.system, name).n > 0
    }

    for key, val in owners.items():
        owner = getattr(rtn.system, key)
        idx_v = owner.get_all_idxes()
        val['idx'] = idx_v
        val['name'] = owner.get(src='name', attr='v', idx=idx_v)
        val['header'].append('Name')
        val['data'].append(val['name'])

    return owners
