"""
Module for report generation.
"""
import logging
from collections import OrderedDict
from time import strftime

from andes.io.txt import dump_data
from andes.shared import np
from andes.utils.misc import elapsed

from ams import __version__ as version
from ams.shared import copyright_msg

logger = logging.getLogger(__name__)


def report_info(system) -> list:
    info = list()
    info.append('AMS' + ' ' + version + '\n')
    info.append(f'{copyright_msg}\n\n')
    info.append('AMS comes with ABSOLUTELY NO WARRANTY\n')
    info.append('Case file: ' + str(system.files.case) + '\n')
    info.append('Report time: ' + strftime("%m/%d/%Y %I:%M:%S %p") + '\n\n')
    return info


class Report:
    """
    Report class to store routine analysis reports.
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
            'Regions': system.Region.n,
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
        system = self.system

        text = list()
        header = list()
        row_name = list()
        data = list()

        if not rtn.converged:
            return text, header, row_name, data

        # initialize data section by model
        owners_all = ['Bus', 'Line', 'StaticGen',
                      'PV', 'Slack', 'RenGen',
                      'DG', 'ESD1', 'PVD1']

        # Filter owners that exist in the system
        owners_e = [var.owner.class_name for var in rtn.vars.values() if var.owner is not None]

        # Use a dictionary comprehension to create vars_by_owner
        owners = {
            name: {'idx': [],
                   'name': [],
                   'header': [],
                   'data': [], }
            for name in owners_all if name in owners_e and getattr(system, name).n > 0
        }

        # --- owner data: idx and name ---
        for key, val in owners.items():
            owner = getattr(system, key)
            idx_v = owner.get_idx()
            val['idx'] = idx_v
            val['name'] = owner.get(src='name', attr='v', idx=idx_v)
            val['header'].append('Name')
            val['data'].append(val['name'])

        # --- variables data ---
        for key, var in rtn.vars.items():
            owner_name = var.owner.class_name
            idx_v = owners[owner_name]['idx']
            header_v = key if var.unit is None else f'{key} ({var.unit})'
            data_v = rtn.get(src=key, attr='v', idx=idx_v, horizon=horizon).round(6)
            owners[owner_name]['header'].append(header_v)
            owners[owner_name]['data'].append(data_v)

        # --- dump data ---
        for key, val in owners.items():
            text.append([f'{key} DATA:\n'])
            row_name.append(val['idx'])
            header.append(val['header'])
            data.append(val['data'])
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
                if rtn.type == 'ACED':
                    header.append(['P (p.u.)', 'Q (p.u.)'])
                    Pcol = [rtn.pg.v.sum().round(6), rtn.pd.v.sum().round(6)]
                    Qcol = [rtn.qg.v.sum().round(6), rtn.qd.v.sum().round(6)]
                    data.append([Pcol, Qcol])
                else:
                    header.append(['P (p.u.)'])
                    Pcol = [rtn.pg.v.sum().round(6), rtn.pd.v.sum().round(6)]
                    data.append([Pcol])
                # --- routine data ---
                text.extend(text_sum)
                header.extend(header_sum)
                row_name.extend(row_name_sum)
                data.extend(data_sum)
        dump_data(text, header, row_name, data, system.files.txt)

        _, s = elapsed(t)
        logger.info(f'Report saved to "{system.files.txt}" in {s}.')
