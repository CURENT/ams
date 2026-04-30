"""
Runtime attribute proxy for routine symbols.

Provides the namespace passed to ``e_fn(r)`` callables in the
opt-layer overhaul (Phase 4.1+). Mirrors the 5-category symbol
resolution order used by :class:`ams.core.symprocessor.SymProcessor`
so author-supplied ``e_fn`` bodies see the same names ``e_str``
strings would resolve to under the legacy regex pipeline.

Resolution order (matches ``symprocessor.generate_symbols``):

1. Vars             →  ``cp.Variable`` from ``rtn.<name>.optz``
2. RParams          →  ``cp.Parameter`` from ``rtn.om.<name>``,
                       or sparse-matrix ``_v``,
                       or numpy array (``no_parse``/dense matrix)
3. Routine Services →  ``rtn.om.<name>`` (or ``rtn.<name>.v`` if
                       ``no_parse``)
4. Expressions      →  ``rtn.om.<name>``
5. Constraints      →  ``rtn.<name>.optz`` (for ExpressionCalc-style
                       references)
6. Config           →  ``rtn.config.<key>``

Plus the conventional symbols ``sys_f`` and ``sys_mva``.
"""

import logging

from ams.core.matprocessor import MatProcessor

logger = logging.getLogger(__name__)


class RoutineNS:
    """
    Attribute proxy passed as ``r`` to ``e_fn(r)`` callables.

    Holds a reference to the owning routine and resolves attribute
    access by walking the same five symbol buckets the legacy
    ``sub_map`` regex used. Read-only — there is no setter.
    """

    __slots__ = ('_rtn',)

    def __init__(self, routine):
        # Use object.__setattr__ to avoid recursing through __getattr__.
        object.__setattr__(self, '_rtn', routine)

    def __getattr__(self, name):
        rtn = object.__getattribute__(self, '_rtn')

        # 1. Vars
        vars_ = getattr(rtn, 'vars', {})
        if name in vars_:
            return vars_[name].optz

        # 2. RParams
        rparams = getattr(rtn, 'rparams', {})
        if name in rparams:
            rp = rparams[name]
            if isinstance(rp.owner, MatProcessor):
                if rp.sparse:
                    return getattr(rtn.system.mats, name)._v
                return rp.v
            if rp.no_parse:
                return rp.v
            return getattr(rtn.om, name)

        # 3. Routine Services
        services = getattr(rtn, 'services', {})
        if name in services:
            svc = services[name]
            return svc.v if svc.no_parse else getattr(rtn.om, name)

        # 4. Expressions
        exprs = getattr(rtn, 'exprs', {})
        if name in exprs:
            return getattr(rtn.om, name)

        # 5. Constraints (used by ExpressionCalc-style cross-references)
        constrs = getattr(rtn, 'constrs', {})
        if name in constrs:
            return constrs[name].optz

        # 6. Config
        if hasattr(rtn, 'config') and name in rtn.config.as_dict():
            return getattr(rtn.config, name)

        # 7. Conventional names from BaseRoutine
        if name == 'sys_f':
            return rtn.system.config.freq
        if name == 'sys_mva':
            return rtn.system.config.mva

        raise AttributeError(
            f"RoutineNS: no symbol <{name}> in routine "
            f"<{rtn.class_name}>"
        )

    def __setattr__(self, name, value):
        raise AttributeError(
            "RoutineNS is read-only; attribute assignment is not supported."
        )

    def __repr__(self):
        rtn = object.__getattribute__(self, '_rtn')
        return f'RoutineNS({rtn.class_name})'


class NumericRoutineNS:
    """
    Numeric proxy for ``e_fn(r)``-style callables, used by ``.e``.

    Same resolution order as :class:`RoutineNS`, but returns numpy
    values for Vars (via ``Var.v`` — which falls back to ``np.zeros``
    when the solver hasn't run / bailed) and for cp.Parameter-backed
    RParams (via ``RParam.v``). Expressions are recomputed by
    re-invoking their ``e_fn`` against the same proxy.

    The result of an ``e_fn(numeric_r)`` call is a CVXPY constant
    expression (``cp.multiply`` / ``cp.sum`` etc. wrap numpy inputs in
    ``cp.Constant``). Read ``.value`` (or ``._expr.value`` for a
    Constraint result) to get the numpy LHS — that's what ``.e``
    returns for debugging.
    """

    __slots__ = ('_rtn',)

    def __init__(self, routine):
        object.__setattr__(self, '_rtn', routine)

    def __getattr__(self, name):
        rtn = object.__getattribute__(self, '_rtn')

        # 1. Vars → .v (numpy, zeros fallback when optz.value is None)
        vars_ = getattr(rtn, 'vars', {})
        if name in vars_:
            return vars_[name].v

        # 2. RParams → .v (always a numpy view at the source)
        rparams = getattr(rtn, 'rparams', {})
        if name in rparams:
            rp = rparams[name]
            if isinstance(rp.owner, MatProcessor):
                if rp.sparse:
                    return getattr(rtn.system.mats, name)._v
                return rp.v
            return rp.v

        # 3. Routine Services → .v (numpy)
        services = getattr(rtn, 'services', {})
        if name in services:
            return services[name].v

        # 4. Expressions — recompute against this proxy if e_fn is wired,
        # else fall back to optz.value.
        exprs = getattr(rtn, 'exprs', {})
        if name in exprs:
            expr = exprs[name]
            if expr.e_fn is not None:
                result = expr.e_fn(self)
                return getattr(result, 'value', result)
            return expr.v

        # 5. Constraints — referenced from ExpressionCalc as ``pb.dual_variables[0]``.
        # Return the cvxpy Constraint so ``.dual_variables`` etc. work.
        constrs = getattr(rtn, 'constrs', {})
        if name in constrs:
            return constrs[name].optz

        # 6. Config values
        if hasattr(rtn, 'config') and name in rtn.config.as_dict():
            return getattr(rtn.config, name)

        # 7. Conventional names
        if name == 'sys_f':
            return rtn.system.config.freq
        if name == 'sys_mva':
            return rtn.system.config.mva

        raise AttributeError(
            f"NumericRoutineNS: no symbol <{name}> in routine "
            f"<{rtn.class_name}>"
        )

    def __setattr__(self, name, value):
        raise AttributeError(
            "NumericRoutineNS is read-only; attribute assignment is not supported."
        )

    def __repr__(self):
        rtn = object.__getattribute__(self, '_rtn')
        return f'NumericRoutineNS({rtn.class_name})'
