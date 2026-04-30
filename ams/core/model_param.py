"""
Minimal scaffolding for the Phase 4.1+ parameter wrapper.

Provides ``.v`` (numpy view) and ``.cp`` (cached ``cp.Parameter`` view)
on a single value source. Authors of ``e_fn(r)`` routine bodies pick
``.cp`` for parameters that change between solves (DPP-cached warm
re-solves) and ``.v`` for parameters baked in at build time
(``cp.Constant`` semantics; routine rebuild required to change).

This is the scaffolding form: it *wraps* an existing source (an
``RParam``-like object with a ``.v`` property, or a raw value). The
fuller form that absorbs ``RParam``'s indexer/imodel logic and
replaces ``ams.opt.Param`` lands in Phase 4.7 cleanup.
"""

import logging
from typing import Any, Optional

import cvxpy as cp

logger = logging.getLogger(__name__)


class ModelParam:
    """
    Dual-access wrapper around a parameter source.

    Parameters
    ----------
    source : object, optional
        Any object exposing a ``.v`` property that returns the current
        numerical value (e.g., an :class:`ams.core.RParam`). If
        provided, ``ModelParam.v`` delegates to ``source.v``.
    value : Any, optional
        Raw value (typically a numpy array). Used when no ``source`` is
        supplied. Mutually exclusive with ``source``.
    tex_name : str, optional
        LaTeX name forwarded by routines for documentation generation.
    **cp_attrs
        Forwarded to :class:`cp.Parameter` on first ``.cp`` access
        (e.g., ``nonneg=True``, ``symmetric=True``).
    """

    __slots__ = ('_source', '_value', '_cp', '_cp_attrs', 'tex_name', 'name')

    def __init__(self,
                 source: Optional[Any] = None,
                 value: Optional[Any] = None,
                 tex_name: Optional[str] = None,
                 name: Optional[str] = None,
                 **cp_attrs):
        if source is not None and value is not None:
            raise ValueError("ModelParam: pass exactly one of `source` or `value`.")
        self._source = source
        self._value = value
        self._cp = None
        self._cp_attrs = cp_attrs
        self.tex_name = tex_name
        self.name = name

    @property
    def v(self):
        """Current numerical value (delegates to source or returns stored value)."""
        if self._source is not None:
            return self._source.v
        return self._value

    @property
    def cp(self):
        """Cached :class:`cp.Parameter`, lazily constructed and pushed.

        First access creates the parameter sized to the current ``.v``
        shape and seeds its ``value``. Subsequent calls return the
        same instance — that's the whole point: holding identity lets
        DPP cache the canonicalization across re-solves.
        """
        if self._cp is None:
            arr = self.v
            shape = getattr(arr, 'shape', ())
            self._cp = cp.Parameter(shape=shape, **self._cp_attrs)
            self._cp.value = arr
        return self._cp

    def push(self):
        """Push current ``.v`` into the cached ``cp.Parameter`` (no-op if unbuilt)."""
        if self._cp is not None:
            self._cp.value = self.v

    def __repr__(self):
        kind = 'source' if self._source is not None else 'value'
        return f'ModelParam({self.name!r}, {kind})'
