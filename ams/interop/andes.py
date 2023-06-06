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

from ams.io import input_formats, guess

logger = logging.getLogger(__name__)


def to_andes(system,
             setup=True,
             addfile=None,
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

    df_models = system.as_dict()

    na_models = []
    # Initialize an ANDES system
    sa = andes_system(**kwargs)
    sa.files.ext = '.xlsx'  # pseudo xlsx extension
    for name, df in df_models.items():
        # convert dict to dataframe
        df = pd.DataFrame(df).drop(['uid'], axis=1)
        try:
            andes_mdl = getattr(sa, name)
        except AttributeError:
            na_models.append(name)
            continue
        andes_params = list(andes_mdl.params.keys())
        columns_to_drop = list(set(df.columns) - set(andes_params))
        df = pd.DataFrame(df).drop(columns_to_drop,
                                   axis=1)
        # drop rows that all nan
        df.dropna(axis=0, how='all', inplace=True)
        for row in df.to_dict(orient='records'):
            sa.add(name, row)

    _, s = elapsed(t0)
    logger.info('System convert to ANDES in %s.', s)

    if na_models:
        logger.debug(f'No corresponding model found in ANDES: {na_models}')

    # Try parsing the addfile
    t, _ = elapsed()

    # additonal file for dynamic simulation
    if addfile:
        sa.files.addfile = addfile
        sa.files.add_format = guess(sa)
        logger.info('Parsing additional file "%s"...', sa.files.addfile)
        add_parser = importlib.import_module('andes.io.' + sa.files.add_format)
        try:
            add_parser.read_add(sa, sa.files.addfile)
            # logger.error('Error parsing addfile "%s" with %s parser.', sa.files.addfile, input_format)
        except AttributeError:
            add_parser.read(sa, addfile)

        # check SynGen Vn
        syngen_idx = sa.SynGen.find_idx(keys='bus', values=sa.Bus.idx.v, allow_none=True, default=None)
        syngen_idx = [x for x in syngen_idx if x is not None]
        syngen_bus = sa.SynGen.get(src='bus', idx=syngen_idx, attr='v')

        gen_vn = sa.SynGen.get(src='Vn', idx=syngen_idx, attr='v')
        bus_vn = sa.Bus.get(src='Vn', idx=syngen_bus, attr='v')
        if not np.equal(gen_vn, bus_vn).all():
            sa.SynGen.set(src='Vn', idx=syngen_idx, attr='v', value=bus_vn)
            logger.warning('Correct SynGen Vn to match bus Vn.')
        _, s = elapsed(t)
        logger.info('Addfile parsed in %s.', s)

    if setup:
        sa.setup()

    return sa
