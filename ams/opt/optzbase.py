"""
Module for optimization base classes.
"""
import logging

from typing import Optional

import cvxpy as cp

from ams.utils.misc import deprec_get_idx


logger = logging.getLogger(__name__)


def ensure_symbols(func):
    """
    Decorator to ensure that symbols are generated before parsing.
    If not, it runs self.rtn.syms.generate_symbols().

    Designed to be used on the `parse` method of the optimization elements (`OptzBase`)
    and optimization model (`OModel`), i.e., `Var`, `Param`, `Constraint`, `Objective`,
    and `ExpressionCalc`.

    Parsing before symbol generation can give wrong results. Ensure that symbols
    are generated before calling the `parse` method.
    """

    def wrapper(self, *args, **kwargs):
        if not self.rtn._syms:
            logger.debug(f"<{self.rtn.class_name}> symbols are not generated yet. Generating now...")
            self.rtn.syms.generate_symbols()
        return func(self, *args, **kwargs)
    return wrapper


def ensure_mats_and_parsed(func):
    """
    Decorator to ensure that system matrices are built and the OModel is parsed
    before evaluation. If not, it runs the necessary methods to initialize them.

    Designed to be used on the `evaluate` method of optimization elements (`OptzBase`)
    and the optimization model (`OModel`), i.e., `Var`, `Param`, `Constraint`, `Objective`,
    and `ExpressionCalc`.

    Evaluation before building matrices and parsing the OModel can lead to errors. Ensure that
    system matrices are built and the OModel is parsed before calling the `evaluate` method.
    """

    def wrapper(self, *args, **kwargs):
        try:
            if not self.rtn.system.mats.initialized:
                logger.debug("System matrices are not built yet. Building now...")
                self.rtn.system.mats.build()
            if isinstance(self, (OptzBase)):
                if not self.om.parsed:
                    logger.debug("OModel is not parsed yet. Parsing now...")
                    self.om.parse()
            else:
                if not self.parsed:
                    logger.debug("OModel is not parsed yet. Parsing now...")
                    self.parse()
        except Exception as e:
            logger.error(f"Error during initialization or parsing: {e}")
            raise e
        return func(self, *args, **kwargs)
    return wrapper


class _EFormDescriptor:
    """Mutex descriptor for ``e_str`` / ``e_fn`` on opt elements.

    Setting one to a non-None value clears the other so the most recent
    assignment wins. Lets subclasses override an inherited element's
    ``e_str`` without inheriting a stale ``e_fn`` from the parent class.

    When the assignment *replaces* a previously-set other form, set the
    ``_e_dirty`` flag on the instance. ``_link_pycode`` reads this flag to
    decide whether to wire codegen output: a dirty item has been
    user-modified at runtime (e.g. ``obj.e_str += '...'``) and must keep
    its current state; auto-prep would otherwise overwrite the user's
    intent with a stale callable from the disk cache. Codegen wiring
    itself bypasses the descriptor by writing ``_e_fn`` directly, so it
    never trips this flag.
    """

    def __init__(self, mine, other):
        self._mine = '_' + mine
        self._other = '_' + other

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        return getattr(obj, self._mine, None)

    def __set__(self, obj, value):
        prior_other = getattr(obj, self._other, None)
        setattr(obj, self._mine, value)
        if value is not None:
            setattr(obj, self._other, None)
            if prior_other is not None:
                obj._e_dirty = True
            # Update provenance. Public-API ``e_fn = fn`` is "manual";
            # ``e_str = '...'`` clears the provenance because the live
            # e_fn (if any) just got cleared by the line above.
            if self._mine == '_e_fn':
                obj._e_fn_source = 'manual'
            else:
                obj._e_fn_source = None


class OptzBase:
    """
    Base class for optimization elements.
    Ensure that symbols are generated before calling the `parse` method. Parsing
    before symbol generation can lead to incorrect results.

    Parameters
    ----------
    name : str, optional
        Name of the optimization element.
    info : str, optional
        Descriptive information about the optimization element.
    unit : str, optional
        Unit of measurement for the optimization element.

    Attributes
    ----------
    rtn : ams.routines.Routine
        The owner routine instance.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 model: Optional[str] = None,
                 ):
        self.om = None
        self.name = name
        self.info = info
        self.unit = unit
        self.is_disabled = False
        self.rtn = None
        self.optz = None  # corresponding optimization element
        self.code = None
        # Pre-rendered LaTeX, populated by ``RoutineBase._link_pycode``
        # from the generator's ``_<prefix>_<name>_tex`` strings. The
        # documenter prefers this over running ``tex_map`` regex at
        # doc-build time, which lets LaTeX rendering survive the
        # descriptor mutex clearing ``e_str`` after ``e_fn`` is wired.
        self.e_tex = None
        self.model = model  # indicate if this element belongs to a model or group
        self.owner = None  # instance of the owner model or group
        self.is_group = False
        # Provenance of the live ``e_fn`` (when one is set). Values:
        #   - 'codegen' : wired from ``~/.ams/pycode/`` by ``_link_pycode``
        #   - 'manual'  : assigned directly via ``item.e_fn = fn``
        #   - None      : no e_fn (uses the eval-fallback helper)
        # Read via :pyattr:`formulation_source` — this raw attribute is
        # written by ``_link_pycode`` and reset whenever ``e_str`` is
        # reassigned (descriptor mutex side-effect).
        self._e_fn_source = None

    @ensure_symbols
    def parse(self):
        """
        Parse the object.
        """
        raise NotImplementedError

    @ensure_mats_and_parsed
    def evaluate(self):
        """
        Evaluate the object.
        """
        raise NotImplementedError

    @property
    def class_name(self):
        """
        Return the class name
        """
        return self.__class__.__name__

    @property
    def formulation_source(self):
        """
        Where the live CVXPY object for this item came from. Useful for
        debugging "did my customization actually take effect?".

        Returns one of:

        - ``'pending'`` — item hasn't been evaluated yet (no live ``optz``).
        - ``'codegen'`` — ``e_fn`` was wired from ``~/.ams/pycode/<routine>.py``
          (the fast AOT path); only happens for items that exactly match
          the pristine source.
        - ``'manual'`` — author/user code assigned ``item.e_fn = fn``
          directly, bypassing both codegen and the eval-fallback helper.
        - ``'eval'`` — eval-fallback path: ``e_str`` is resolved
          symbol-by-symbol via :func:`ams.opt._runtime_eval.eval_e_str`
          and ``eval``-ed at parse/evaluate time. This is what runs for
          items the user customized via ``e_str = '...'`` or
          ``addConstrs(...)``. (Renamed from ``'sub_map'`` in v1.2.2;
          the legacy regex+eval pipeline that name referenced was
          retired in PR #246.)
        """
        if getattr(self, 'optz', None) is None:
            return 'pending'
        if getattr(self, '_e_fn_source', None) == 'codegen':
            return 'codegen'
        if getattr(self, '_e_fn_source', None) == 'manual':
            return 'manual'
        if getattr(self, 'e_fn', None) is not None:
            # e_fn set but provenance lost — defensive fallback. Shouldn't
            # happen in normal flow.
            return 'manual'
        return 'eval'

    @property
    def n(self):
        """
        Return the number of elements.
        """
        if self.owner is None:
            return len(self.v)
        else:
            return self.owner.n

    @property
    def shape(self):
        """
        Return the shape.
        """
        try:
            return self.om.__dict__[self.name].shape
        except KeyError:
            logger.warning('Shape info is not ready before initialization.')
            return None

    @property
    def size(self):
        """
        Return the size.
        """
        if self.rtn.initialized:
            return self.om.__dict__[self.name].size
        else:
            logger.warning(f'Routine <{self.rtn.class_name}> is not initialized yet.')
            return None

    @property
    def e(self):
        """
        Return the calculated numerical value of the underlying expression.

        Used for debugging — for a successfully solved problem, ``e`` should
        equal ``v``. For infeasible/unbounded problems, ``e`` lets you
        inspect the LHS at the returned (possibly invalid) point.

        Two paths:

        - **e_fn form**: ``self.code`` is ``None`` because no string was
          parsed; fall back to the cvxpy object's value (``self.optz._expr``
          for constraints, ``self.optz`` otherwise). This is the
          solver-returned value, not a re-canonicalization from current
          parameter state — sufficient for the post-solve ``v == e`` check.
        - **e_str form**: route through :func:`eval_e_str_numeric`,
          which mirrors the symbol-resolution surface of the CVXPY-side
          helper but resolves to numpy via :class:`NumericRoutineNS`.
        """
        if self.code is None:
            # e_fn form: re-evaluate against the current numeric state
            # via NumericRoutineNS. Resolves Vars to ``Var.v`` (which
            # falls back to ``np.zeros`` when ``optz.value`` is None),
            # so ``.e`` is informative even on a failed/incomplete
            # solve — matching the legacy val_map+eval behavior.
            #
            # The e_fn may return: (a) a cp.Constraint (legacy form),
            # (b) a cp.Expression (LHS or Objective inner), or (c) a
            # pure numpy result when every operand is numpy and no
            # cp.X wrapper is present. Handle all three.
            from ams.core.routine_ns import NumericRoutineNS
            e_fn = getattr(self, 'e_fn', None)
            if e_fn is not None:
                try:
                    result = e_fn(NumericRoutineNS(self.om.rtn))
                except Exception as exc:
                    logger.error(
                        f"Error in re-evaluating {self.class_name} "
                        f"<{self.name}> via e_fn for `.e`.\n{exc}"
                    )
                    return None
                # Constraint legacy form: extract LHS via _expr.value.
                if isinstance(result, cp.constraints.Constraint):
                    inner = getattr(result, '_expr', None)
                    if inner is None:
                        # Some constraint subclasses expose ``args[0]``
                        inner = result.args[0] if result.args else None
                    return getattr(inner, 'value', None)
                # cp.Expression / cp.Constant: take .value.
                if hasattr(result, 'value'):
                    return result.value
                # Pure numpy: that IS the value.
                return result

            # No e_fn either — fall back to cached optz.
            optz = getattr(self, 'optz', None)
            if optz is None:
                logger.info(f"{self.class_name} <{self.name}> is not evaluated yet.")
                return None
            inner = getattr(optz, '_expr', optz)
            return getattr(inner, 'value', None)

        # e_str form: defer to the numeric helper. Same name-resolution
        # surface as the CVXPY-side ``eval_e_str``, but values come from
        # ``NumericRoutineNS`` so the result is a ``cp.Constant`` whose
        # ``.value`` is the numpy LHS.
        from ams.opt._runtime_eval import eval_e_str_numeric
        try:
            result = eval_e_str_numeric(self, self.code)
        except Exception as exc:
            logger.error(f"Error in calculating {self.class_name} <{self.name}>.\n{exc}")
            return None
        return getattr(result, 'value', result)

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.name}'

    @deprec_get_idx
    def get_idx(self):
        if self.is_group:
            return self.owner.get_all_idxes()
        elif self.owner is None:
            logger.info(f'{self.class_name} <{self.name}> has no owner.')
            return None
        else:
            return self.owner.idx.v

    def get_all_idxes(self):
        """
        Return all the indexes of this item.

        Returns
        -------
        list
            A list of indexes.

        Notes
        -----
        .. versionadded:: 1.0.0
        """

        if self.is_group:
            return self.owner.get_all_idxes()
        elif self.owner is None:
            logger.info(f'{self.class_name} <{self.name}> has no owner.')
            return None
        else:
            return self.owner.idx.v
