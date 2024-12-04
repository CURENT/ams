"""
Module for plotting functions.
"""

import logging

from andes.shared import plt

from ams.shared import np

logger = logging.getLogger(__name__)
DPI = None


class Plotter:
    """
    Class for routine plotting functions.
    """

    def __init__(self, system):
        self.system = system

    def bar(self, yidx, xidx=None, *, a=None, ytimes=None, ycalc=None,
            left=None, right=None, ymin=None, ymax=None,
            xlabel=None, ylabel=None, xheader=None, yheader=None,
            legend=None, grid=False, greyscale=False, latex=True,
            dpi=DPI, line_width=1.0, font_size=12, savefig=None, save_format=None, show=True,
            title=None, linestyles=None,
            hline1=None, hline2=None, vline1=None, vline2=None, hline=None, vline=None,
            fig=None, ax=None,
            set_xlim=True, set_ylim=True, autoscale=False,
            legend_bbox=None, legend_loc=None, legend_ncol=1,
            figsize=None, color=None,
            **kwargs):
        """
        Plot the variable in a bar plot.
        """

        if len(yidx) == 0:
            logger.error("No variables to plot.")
            return None

        y_idx = yidx.get_idx()

        if xidx is None:
            xidx = yidx.horizon.v

        x_idx = [yidx.horizon.v.index(x) for x in xidx]

        yvalue = yidx.v[np.ix_(y_idx, x_idx)]

        x_loc = np.arange(len(x_idx))
