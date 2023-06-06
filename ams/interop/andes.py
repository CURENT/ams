"""
Interface with ANDES
"""

import importlib
import logging
import json

from andes.models import file_classes
from andes.system import System as andes_system
from andes.shared import pd, np
from andes.utils.misc import elapsed
from andes import load as andes_load

from ams.io import input_formats, guess, xlsx

logger = logging.getLogger(__name__)


def to_andes(system,
             setup=True,
             addfile=None,
             overwrite=None,
             **kwargs):
    """
    Convert the current model to ANDES format.

    Parameters
    ----------
    system: System
        The current system to be converted to ANDES format.
    setup: bool
        Whether to call `setup()` after the conversion.
    kwargs: dict
        Keyword arguments to be passed to `andes.system.System`.
    """
    t0, _ = elapsed()
    andes_file = system.files.name + '.xlsx'

    xlsx.write(system, andes_file,
               overwrite=overwrite,
               to_andes=True,
               **kwargs,
               )

    _, s = elapsed(t0)
    logger.info(f'System convert to ANDES in {s}, save to "{andes_file}".')

    sa = andes_load(andes_file, setup=setup, addfile=addfile, **kwargs)

    # Try parsing the addfile
    t, _ = elapsed()

    # additonal file for dynamic simulation
    # if addfile:
    #     sa.files.addfile = addfile
    #     sa.files.add_format = guess(sa)
    #     logger.info('Parsing additional file "%s"...', sa.files.addfile)
    #     add_parser = importlib.import_module('andes.io.' + sa.files.add_format)
    #     try:
    #         add_parser.read_add(sa, sa.files.addfile)
    #         # logger.error('Error parsing addfile "%s" with %s parser.', sa.files.addfile, input_format)
    #     except AttributeError:
    #         add_parser.read(sa, addfile)

    #     # check SynGen Vn
    #     syngen_idx = sa.SynGen.find_idx(keys='bus', values=sa.Bus.idx.v, allow_none=True, default=None)
    #     syngen_idx = [x for x in syngen_idx if x is not None]
    #     syngen_bus = sa.SynGen.get(src='bus', idx=syngen_idx, attr='v')

    #     gen_vn = sa.SynGen.get(src='Vn', idx=syngen_idx, attr='v')
    #     bus_vn = sa.Bus.get(src='Vn', idx=syngen_bus, attr='v')
    #     if not np.equal(gen_vn, bus_vn).all():
    #         sa.SynGen.set(src='Vn', idx=syngen_idx, attr='v', value=bus_vn)
    #         logger.warning('Correct SynGen Vn to match bus Vn.')
    #     _, s = elapsed(t)
    #     logger.info('Addfile parsed in %s.', s)

    return sa
