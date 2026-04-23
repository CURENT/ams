"""
Table formatting utilities for AMS documentation output.

Notes
-----
Adapted from ``andes.utils.tab``.
Original author: Hantao Cui. License: GPL-3.0.
"""

from texttable import Texttable


class Tab(Texttable):
    """
    Use package ``texttable`` to create well-formatted tables for setting and
    device documentation.

    Parameters
    ----------
    export : {'plain', 'rest'}
        Export format in plain text or restructuredText.
    max_width : int
        Maximum table width. Set to 0 to disable wrapping (useful when cells
        contain equations).
    """

    def __init__(self,
                 title=None,
                 header=None,
                 descr=None,
                 data=None,
                 export='plain',
                 max_width=78):
        Texttable.__init__(self, max_width=max_width)
        if export == 'plain':
            self.set_chars(['-', '|', '+', '-'])
            self.set_deco(Texttable.HEADER | Texttable.VLINES)

        self._title = title
        self._descr = descr
        if header is not None:
            self.header(header)
        if data is not None:
            self.add_rows(data, header=False)

    def header(self, header_list):
        """Set the header with a list."""
        Texttable.header(self, header_list)

    def set_title(self, val):
        """Set table title to ``val``."""
        self._title = val

    def _add_left_space(self, nspace=1):
        """Add *nspace* columns of spaces before the first column."""
        sp = ' ' * nspace
        for item in self._rows:
            item[0] = sp + item[0]

    def draw(self):
        """Draw the table and return it as a string."""
        self._add_left_space()

        if self._title and self._descr:
            pre = self._title + '\n' + self._descr + '\n\n'
        elif self._title:
            pre = self._title + '\n\n'
        elif self._descr:
            pre = 'Empty Title' + '\n' + self._descr + '\n'
        else:
            pre = ''
        return pre + str(Texttable.draw(self)) + '\n\n'


def make_doc_table(title, max_width, export, plain_dict, rest_dict):
    """
    Format documentation data into a :class:`Tab` table.

    Notes
    -----
    Adapted from ``andes.utils.tab.make_doc_table``.
    Original author: Hantao Cui. License: GPL-3.0.
    """
    data_dict = rest_dict if export == 'rest' else plain_dict

    table = Tab(title=title, max_width=max_width, export=export)
    table.header(list(data_dict.keys()))

    rows = list(map(list, zip(*list(data_dict.values()))))
    table.add_rows(rows, header=False)

    return table.draw()


def math_wrap(tex_str_list, export):
    """
    Wrap each item in *tex_str_list* with a LaTeX math environment ``$...$``.

    Only wraps when *export* is ``'rest'``; otherwise returns the list unchanged.

    Notes
    -----
    Adapted from ``andes.utils.tab.math_wrap``.
    Original author: Hantao Cui. License: GPL-3.0.
    """
    if export != 'rest':
        return list(tex_str_list)

    out = []
    for item in tex_str_list:
        if item is None or item == '':
            out.append('')
        else:
            out.append(rf':math:`{item}`')
    return out
