"""
Module for optimization modeling.
"""

from ams.opt.optzbase import OptzBase, ensure_symbols, ensure_mats_and_parsed  # NOQA
from ams.opt.var import Var  # NOQA
# NB: ExpressionCalc must be imported before Param so that Param's importers
# (matprocessor) don't pull a half-loaded opt namespace. ExprCalc itself
# defers `from ams.core.routine_ns import RoutineNS` to method scope to
# break the resulting cycle.
from ams.opt.exprcalc import ExpressionCalc  # NOQA
from ams.opt.param import Param  # NOQA
from ams.opt.constraint import Constraint  # NOQA
from ams.opt.objective import Objective  # NOQA
from ams.opt.expression import Expression  # NOQA
from ams.opt.omodel import OModelBase, OModel  # NOQA

__all__ = [
    'OptzBase', 'ensure_symbols', 'ensure_mats_and_parsed',
    'Var', 'ExpressionCalc', 'Param', 'Constraint',
    'Objective', 'Expression', 'OModelBase', 'OModel',
]
