"""
Module for routine data.
"""

import logging
import json
from collections import OrderedDict
from typing import Optional, Union, Type, Iterable, Dict

import numpy as np

from ams.utils.misc import elapsed

from ams.core import Config
from ams.core.param import RParam
from ams.core.symprocessor import SymProcessor
from ams.core.documenter import RDocumenter
from ams.core.service import RBaseService, ValueService
from ams.opt import OModel
from ams.opt import Param, Var, Constraint, Objective, ExpressionCalc, Expression

from ams.utils.paths import get_export_path

from ams.shared import pd, summary_row, summary_name

logger = logging.getLogger(__name__)


class RoutineBase:
    """
    Class to hold descriptive routine models and data mapping.

    Attributes
    ----------
    system : Optional[Type]
        The system object associated with the routine.
    config : Config
        Configuration object for the routine.
    info : Optional[str]
        Information about the routine.
    tex_names : OrderedDict
        LaTeX names for the routine parameters.
    syms : SymProcessor
        Symbolic processor for the routine.
    _syms : bool
        Flag indicating whether symbols have been generated.
    rparams : OrderedDict
        Registry for RParam objects.
    services : OrderedDict
        Registry for service objects.
    params : OrderedDict
        Registry for Param objects.
    vars : OrderedDict
        Registry for Var objects.
    constrs : OrderedDict
        Registry for Constraint objects.
    exprcs : OrderedDict
        Registry for ExpressionCalc objects.
    exprs : OrderedDict
        Registry for Expression objects.
    obj : Optional[Objective]
        Objective of the routine.
    initialized : bool
        Flag indicating whether the routine has been initialized.
    type : str
        Type of the routine.
    docum : RDocumenter
        Documentation generator for the routine.
    map1 : OrderedDict
        Mapping from ANDES.
    map2 : OrderedDict
        Mapping to ANDES.
    om : OModel
        Optimization model for the routine.
    exec_time : float
        Execution time of the routine.
    exit_code : int
        Exit code of the routine.
    converged : bool
        Flag indicating whether the routine has converged.
    converted : bool
        Flag indicating whether AC conversion has been performed.
    """

    def __init__(self, system=None, config=None, **kwargs):
        """
        Initialize the routine.

        Parameters
        ----------
        system : Optional[Type]
            The system object associated with the routine.
        config : Optional[dict]
            Configuration dictionary for the routine.
        """
        self.system = system
        self.config = Config(self.class_name)
        self.tex_names = OrderedDict(
            (
                ("sys_f", "f_{sys}"),
                ("sys_mva", "S_{b,sys}"),
            )
        )
        self.syms = SymProcessor(self)      # symbolic processor
        self._syms = False                  # symbol generation flag

        self.rparams = OrderedDict()        # RParam registry
        self.services = OrderedDict()       # Service registry
        self.params = OrderedDict()         # Param registry
        self.vars = OrderedDict()           # Var registry
        self.constrs = OrderedDict()        # Constraint registry
        self.exprcs = OrderedDict()         # ExpressionCalc registry
        self.exprs = OrderedDict()          # Expression registry
        self.obj = None                     # Objective
        self.initialized = False            # initialization flag
        self.type = "UndefinedType"         # routine type
        self.docum = RDocumenter(self)      # documentation generator

        # --- sync mapping ---
        self.map1 = OrderedDict()  # from ANDES
        self.map2 = OrderedDict()  # to ANDES

        # --- optimization modeling ---
        self.om = OModel(routine=self)      # optimization model

        if config is not None:
            self.config.load(config)

        # NOTE: the difference between exit_code and converged is that
        # exit_code is the solver exit code, while converged is the
        # convergence flag of the routine.
        self.exec_time = 0.0        # running time
        self.exit_code = 0          # exit code
        self.converged = False      # convergence flag
        self.converted = False          # AC conversion flag

    @property
    def class_name(self):
        return self.__class__.__name__

    def _link_pycode(self):
        """
        Ensure pycode for this routine is current, then wire generated
        ``e_fn`` callables onto items that don't already have one.

        Idempotent. Called from ``OModel.init`` (the one place where we
        know the routine instance is fully constructed and we're about
        to hit the parse/evaluate path).

        Cache key is an md5 of the routine class's source file plus the
        cvxpy / ams versions; mismatch triggers a regen.
        """
        import importlib.util
        from pathlib import Path

        import cvxpy as _cp

        from ams.prep import (
            _get_pristine_system, generate_for_routine, source_md5,
        )

        target = (Path.home() / '.ams' / 'pycode'
                  / f'{self.class_name.lower()}.py')
        expected_md5 = source_md5(type(self))

        # Resolve the pristine routine instance up front. The wire step
        # below uses it to detect user mutation of e_str even on the
        # cache-hit path (a user can mutate before init while a cache
        # already exists from a prior run).
        sys_p = _get_pristine_system()
        pristine_rtn = getattr(sys_p, self.class_name, None)

        gen = None
        # Try to use existing pycode if it matches.
        if target.exists():
            try:
                spec = importlib.util.spec_from_file_location(
                    f'ams._user_pycode.{self.class_name.lower()}', target)
                gen = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(gen)
                # Staleness conditions:
                # - ``md5`` mismatch — routine source file changed.
                # - ``cvxpy_version`` mismatch — bound CVXPY version differs.
                # - ``pristine`` absent or False — cache was written from a
                #   live (possibly customized) instance by an older AMS
                #   version. Reject it so the regen path below produces a
                #   faithful snapshot of the source.
                # NB: ``ams.__version__`` deliberately *not* part of the
                # check — setuptools-scm gives dev installs a ``.postN+g…``
                # suffix that bumps every commit, which would force regen
                # on every save with no behavior delta.
                stale = (
                    getattr(gen, 'md5', None) != expected_md5
                    or getattr(gen, 'cvxpy_version', None) != _cp.__version__
                    or not getattr(gen, 'pristine', False)
                )
                if stale:
                    gen = None
            except Exception as exc:
                logger.debug(
                    f"pycode at {target} not loadable ({exc}); regenerating."
                )
                gen = None

        # Regenerate if missing or stale. Codegen always runs against a
        # pristine routine pulled from a fresh ``ams.System`` — never
        # against ``self``, which may carry user customizations
        # (``addConstrs``, ``obj.e_str += '...'``). This keeps the disk
        # cache faithful to the routine's source code so that a second
        # ``System`` instance loading the cache later doesn't inherit the
        # first user's mutations.
        if gen is None:
            target.parent.mkdir(parents=True, exist_ok=True)
            codegen_src_rtn = pristine_rtn
            if codegen_src_rtn is None:
                # Defensive: every registered routine class should be on
                # the pristine system. If it isn't, the routine isn't
                # standard — fall back to ``self``.
                logger.debug(
                    f"<{self.class_name}> not present on pristine System; "
                    f"falling back to self for codegen."
                )
                codegen_src_rtn = self
            src = generate_for_routine(codegen_src_rtn)
            target.write_text(src)
            spec = importlib.util.spec_from_file_location(
                f'ams._user_pycode.{self.class_name.lower()}', target)
            gen = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gen)

        # Build a lookup of pristine ``_e_str`` values keyed by (prefix,
        # name) so we can detect when ``self`` has diverged from the
        # source. Divergence implies user customization (either pre-init
        # ``obj.e_str = ...`` or post-init ``obj.e_str += ...``); in
        # either case we must NOT wire the codegen callable, or it would
        # override the user's intent.
        def _pristine_e_str(prefix, name):
            if pristine_rtn is None or pristine_rtn is self:
                return None
            registry = {
                'expr':     pristine_rtn.exprs,
                'constr':   pristine_rtn.constrs,
                'exprcalc': pristine_rtn.exprcs,
            }.get(prefix)
            if registry is not None:
                p_item = registry.get(name)
                return getattr(p_item, '_e_str', None) if p_item else None
            if prefix == 'obj':
                p_obj = pristine_rtn.obj
                if p_obj is not None and p_obj.name == name:
                    return getattr(p_obj, '_e_str', None)
            return None

        # Wire e_fn (and pre-rendered tex) from the generated module onto
        # items missing them. Two semantic notes:
        #
        # 1) We write the raw ``_e_fn`` slot (bypassing the descriptor
        #    mutex) so the original ``e_str`` is preserved on the item.
        #    This lets users do ``routine.obj.e_str += '...'`` post-init
        #    — a documented customization pattern (see examples/ex8.ipynb)
        #    that the previous mutex-clearing behavior broke.
        #
        # 2) We skip items the user has modified relative to the pristine
        #    source. This is detected via ``_e_dirty`` (set by the
        #    descriptor mutex when user replaces a wired e_fn) OR by
        #    direct e_str comparison against the pristine instance
        #    (catches pre-init mutations that don't trip the mutex's
        #    prior-other check). Skipped items flow through the legacy
        #    regex+eval path in ``parse()`` / ``evaluate()``.
        # Tally for the end-of-link summary log. Classifies each item by
        # the runtime path it will take, which mirrors the per-item
        # ``formulation_source`` property.
        tally = {'codegen': 0, 'manual': 0,
                 'sub_map_dirty': 0, 'sub_map_added': 0}

        def _wire(item, prefix, name):
            if getattr(item, '_e_dirty', False):
                # User modified this item; we leave it alone. The runtime
                # path is 'manual' if a callable is now in place,
                # otherwise the legacy 'sub_map' regex pipeline.
                if getattr(item, '_e_fn', None) is not None:
                    tally['manual'] += 1
                else:
                    tally['sub_map_dirty'] += 1
                return
            pristine_str = _pristine_e_str(prefix, name)
            if (pristine_str is not None
                    and getattr(item, '_e_str', None) != pristine_str):
                item._e_dirty = True
                tally['sub_map_dirty'] += 1
                return
            fn = getattr(gen, f'_{prefix}_{name}', None)
            if fn is not None:
                if getattr(item, '_e_fn', None) is None:
                    item._e_fn = fn
                    # Provenance: this e_fn came from disk pycode.
                    item._e_fn_source = 'codegen'
                # Whether we just wired or it was already wired from a
                # previous init, the item's runtime path is the codegen
                # callable.
                tally['codegen'] += 1
            else:
                # Item has no entry in pycode (e.g. added at runtime via
                # ``addConstrs``). It will fall through to the sub_map
                # path at parse/evaluate time.
                tally['sub_map_added'] += 1
            tex = getattr(gen, f'_{prefix}_{name}_tex', None)
            if tex is not None and getattr(item, 'e_tex', None) is None:
                item.e_tex = tex

        for name, expr in self.exprs.items():
            _wire(expr, 'expr', name)
        for name, constr in self.constrs.items():
            _wire(constr, 'constr', name)
        if self.obj is not None:
            _wire(self.obj, 'obj', self.obj.name)
        for name, exprc in self.exprcs.items():
            _wire(exprc, 'exprcalc', name)

        # One info-level log line per init() that summarizes which
        # execution path each item will take. Lets users (and the tests
        # in icebar/ex8/) verify customization actually takes effect by
        # eye, without having to introspect ``formulation_source`` on
        # each item.
        total = sum(tally.values())
        if total > 0:
            parts = [f"codegen={tally['codegen']}/{total}"]
            if tally['manual']:
                parts.append(f"manual={tally['manual']}")
            if tally['sub_map_dirty']:
                parts.append(f"sub_map(customized)={tally['sub_map_dirty']}")
            if tally['sub_map_added']:
                parts.append(f"sub_map(added)={tally['sub_map_added']}")
            logger.info(
                f"<{self.class_name}> formulation: " + ", ".join(parts)
            )

    def formulation_summary(self, return_rows: bool = False):
        """
        Print (or return) a per-item table of the live formulation source.

        Useful for verifying which path each opt element runs through after
        custom edits — e.g. ``addConstrs`` and ``obj.e_str += '...'``
        should show ``sub_map``, while untouched items show ``codegen``.

        Parameters
        ----------
        return_rows : bool, optional
            If True, return the list of ``(kind, name, source, e_str_excerpt)``
            tuples instead of printing. Default False (prints).

        See Also
        --------
        ams.opt.OptzBase.formulation_source : per-item source string.
        """
        rows = []
        for kind, registry in (('expr', self.exprs),
                               ('constr', self.constrs),
                               ('exprcalc', self.exprcs)):
            for name, item in registry.items():
                src = getattr(item, 'formulation_source', '?')
                e_str = getattr(item, '_e_str', None) or ''
                rows.append((kind, name, src, e_str[:60]))
        if self.obj is not None:
            obj = self.obj
            rows.append(('obj', obj.name, getattr(obj, 'formulation_source', '?'),
                         (getattr(obj, '_e_str', None) or '')[:60]))

        if return_rows:
            return rows

        if not rows:
            print(f"<{self.class_name}>: no opt elements registered.")
            return

        kw = max(len(r[0]) for r in rows)
        nw = max(len(r[1]) for r in rows)
        sw = max(len(r[2]) for r in rows)
        print(f"<{self.class_name}> formulation summary "
              f"({sum(1 for r in rows if r[2] == 'codegen')} codegen / "
              f"{sum(1 for r in rows if r[2] == 'sub_map')} sub_map / "
              f"{sum(1 for r in rows if r[2] == 'manual')} manual / "
              f"{sum(1 for r in rows if r[2] == 'pending')} pending)")
        print(f"  {'kind':<{kw}}  {'name':<{nw}}  {'source':<{sw}}  e_str")
        print(f"  {'-'*kw}  {'-'*nw}  {'-'*sw}  {'-'*40}")
        for kind, name, src, e_str in rows:
            print(f"  {kind:<{kw}}  {name:<{nw}}  {src:<{sw}}  {e_str}")

    def get(self, src: str, idx, attr: str = 'v',
            horizon: Optional[Union[int, str, Iterable]] = None):
        """
        Get the value of a variable or parameter.

        Parameters
        ----------
        src: str
            Name of the variable or parameter.
        idx: int, str, or list
            Index of the variable or parameter.
        attr: str
            Attribute name.
        horizon: list, optional
            Horizon index.
        """
        if src not in self.__dict__.keys():
            raise ValueError(f"<{src}> does not exist in <<{self.class_name}>.")
        item = self.__dict__[src]

        if not hasattr(item, attr):
            raise ValueError(f"{attr} does not exist in {self.class_name}.{src}.")

        idx_all = item.get_all_idxes()

        if idx_all is None:
            raise ValueError(f"<{self.class_name}> item <{src}> has no idx.")

        is_format = False  # whether the idx is formatted as a list
        idx_u = None
        if isinstance(idx, (str, int)):
            idx_u = [idx]
            is_format = True
        elif isinstance(idx, (np.ndarray, pd.Series)):
            idx_u = idx.tolist()
        elif isinstance(idx, list):
            idx_u = idx.copy()

        loc = [idx_all.index(idxe) if idxe in idx_all else None for idxe in idx_u]
        if None in loc:
            idx_none = [idxe for idxe in idx_u if idxe not in idx_all]
            msg = f"Var <{self.class_name}.{src}> does not contain value with idx={idx_none}"
            raise ValueError(msg)
        out = getattr(item, attr)[loc]

        if horizon is not None:
            if item.horizon is None:
                raise ValueError(f"horizon is not defined for {self.class_name}.{src}.")
            horizon_all = item.horizon.get_all_idxes()
            if not isinstance(horizon, list):
                raise TypeError(f"horizon must be a list, not {type(horizon)}.")
            loc_h = [
                horizon_all.index(idxe) if idxe in horizon_all else None
                for idxe in horizon
            ]
            if None in loc_h:
                idx_none = [idxe for idxe in horizon if idxe not in horizon_all]
                msg = f"Var <{self.class_name}.{src}> does not contain horizon with idx={idx_none}"
                raise ValueError(msg)
            out = out[:, loc_h]
            if out.shape[1] == 1:
                out = out[:, 0]

        return out[0] if is_format else out

    def set(self, src: str, idx, attr: str = "v", value=0.0):
        """
        Set the value of an attribute of a routine parameter.

        Performs ``self.<src>.<attr>[idx] = value``. This method will not modify
        the input values from the case file that have not been converted to the
        system base. As a result, changes applied by this method will not affect
        the dumped case file.

        To alter parameters and reflect it in the case file, use :meth:`alter`
        instead.

        Parameters
        ----------
        src : str
            Name of the model property
        idx : str, int, float, array-like
            Indices of the devices
        attr : str, optional, default='v'
            The internal attribute of the property to get.
            ``v`` for values, ``a`` for address, and ``e`` for equation value.
        value : array-like
            New values to be set

        Returns
        -------
        bool
            True when successful.
        """
        if self.__dict__[src].owner is not None:
            # TODO: fit to `_v` type param in the future
            owner = self.__dict__[src].owner
            src0 = self.__dict__[src].src
            try:
                res = owner.set(src=src0, idx=idx, attr=attr, value=value)
                return res
            except KeyError as e:
                msg = f"Failed to set <{src0}> in <{owner.class_name}>. "
                msg += f"Original error: {e}"
                raise KeyError(msg)
        else:
            # FIXME: add idx for non-grouped variables
            raise TypeError(f"Variable {self.name} has no owner.")

    def doc(self, max_width=78, export="plain"):
        """
        Retrieve routine documentation as a string.
        """
        return self.docum.get(max_width=max_width, export=export)

    def _get_off_constrs(self):
        """
        Chcek if constraints are turned off.
        """
        disabled = []
        for cname, c in self.constrs.items():
            if c.is_disabled:
                disabled.append(cname)
        if len(disabled) > 0:
            msg = "Disabled constraints: "
            d_str = [f'{constr}' for constr in disabled]
            msg += ", ".join(d_str)
            logger.warning(msg)
        return disabled

    def _data_check(self):
        """
        Check if data is valid for a routine.
        """
        logger.info(f"Entering data check for <{self.class_name}>")
        no_input = []
        owner_list = []
        for rname, rparam in self.rparams.items():
            if rparam.owner is not None:
                # NOTE: skip checking Shunt.g
                if (rparam.owner.class_name == 'Shunt') and (rparam.src == 'g'):
                    pass
                # NOTE: below is special case for PTDF availability check
                elif rparam.owner.class_name == 'MatProcessor':
                    if rparam.src == 'PTDF':
                        if self.system.mats.PTDF._v is None:
                            logger.warning("PTDF is not available, build it now")
                            self.system.mats.build_ptdf()
                elif rparam.owner.n == 0:
                    no_input.append(rname)
                    owner_list.append(rparam.owner.class_name)
                else:
                    # do value check for non-empty rparams
                    if rparam.config.pos:
                        if not np.all(rparam.v > 0):
                            logger.warning(f"RParam <{rname}> should have all positive values.")
                    if rparam.config.neg:
                        if not np.all(rparam.v < 0):
                            logger.warning(f"RParam <{rname}> should have all negative values.")
                    if rparam.config.nonpos:
                        if not np.all(rparam.v <= 0):
                            logger.warning(f"RParam <{rname}> should have all non-positive values.")
                    if rparam.config.nonneg:
                        if not np.all(rparam.v >= 0):
                            logger.warning(f"RParam <{rname}> should have all non-negative values.")

        if len(no_input) > 0:
            logger.error(f"<{self.class_name}> Following models are missing in input: {set(owner_list)}")
            return False

        # TODO: add data validation for RParam, typical range, etc.
        logger.info(" -> Data check passed")
        return True

    def init(self, **kwargs) -> bool:
        """
        Initialize the routine.

        Other parameters
        ----------------
        force: bool
            Whether to force initialization regardless of the current initialization status.
        force_mats: bool
            Whether to force build the system matrices, goes to `self.system.mats.build()`.
        force_constr: bool
            Whether to turn on all constraints.
        force_om: bool
            Whether to force initialize the optimization model.
        """
        force = kwargs.pop('force', False)
        force_mats = kwargs.pop('force_mats', False)
        force_constr = kwargs.pop('force_constr', False)
        force_om = kwargs.pop('force_om', False)

        skip_all = not (force and force_mats) and self.initialized and self.om.initialized

        if skip_all:
            logger.debug(f"{self.class_name} has already been initialized.")
            return True

        t0, _ = elapsed()
        # --- data check ---
        self._data_check()

        # --- turn on all constrs ---
        if force_constr:
            for constr in self.constrs.values():
                constr.is_disabled = False

        # --- matrix build ---
        self.system.mats.build(force=force_mats)

        # --- constraint check ---
        _ = self._get_off_constrs()

        if not self.om.initialized:
            self.om.init(force=force_om)
        _, s_init = elapsed(t0)

        msg = f"<{self.class_name}> "
        if self.om.initialized:
            msg += f"initialized in {s_init}."
            self.initialized = True
        else:
            msg += "initialization failed!"
            self.initialized = False
        logger.info(msg)
        return self.initialized

    def solve(self, **kwargs):
        """
        Solve the routine optimization model.
        """
        raise NotImplementedError

    def unpack(self, res, **kwargs):
        """
        Unpack the results.
        """
        raise NotImplementedError

    def _post_solve(self):
        """
        Post-solve calculations.
        """
        # NOTE: unpack Expressions if owner and arc are available
        for expr in self.exprs.values():
            if expr.owner and expr.src:
                expr.owner.set(src=expr.src, attr='v',
                               idx=expr.get_all_idxes(), value=expr.v)
        return True

    def run(self, **kwargs) -> bool:
        """
        Run the routine.

        Following kwargs go to `self.init()`: `force_init`, `force_mats`, `force_constr`, `force_om`.

        Following kwargs go to `self.solve()`: `solver`, `verbose`, `gp`, `qcp`, `requires_grad`,
        `enforce_dpp`, `ignore_dpp`, `method`, and all rest.

        Parameters
        ----------
        force_init : bool, optional
            If True, force re-initialization. Defaults to False.
        force_mats : bool, optional
            If True, force re-generating matrices. Defaults to False.
        force_constr : bool, optional
            Whether to turn on all constraints.
        force_om : bool, optional
            If True, force re-generating optimization model. Defaults to False.
        solver: str, optional
            The solver to use. For example, 'GUROBI', 'ECOS', 'SCS', or 'OSQP'.
        verbose : bool, optional
            Overrides the default of hiding solver output and prints logging
            information describing CVXPY's compilation process.
        gp : bool, optional
            If True, parses the problem as a disciplined geometric program
            instead of a disciplined convex program.
        qcp : bool, optional
            If True, parses the problem as a disciplined quasiconvex program
            instead of a disciplined convex program.
        requires_grad : bool, optional
            Makes it possible to compute gradients of a solution with respect to Parameters
            by calling problem.backward() after solving, or to compute perturbations to the variables
            given perturbations to Parameters by calling problem.derivative().
            Gradients are only supported for DCP and DGP problems, not quasiconvex problems.
            When computing gradients (i.e., when this argument is True), the problem must satisfy the DPP rules.
        enforce_dpp : bool, optional
            When True, a DPPError will be thrown when trying to solve a
            non-DPP problem (instead of just a warning).
            Only relevant for problems involving Parameters. Defaults to False.
        ignore_dpp : bool, optional
            When True, DPP problems will be treated as non-DPP, which may speed up compilation. Defaults to False.
        method : function, optional
            A custom solve method to use.
        """
        # --- setup check ---
        force_init = kwargs.pop('force_init', False)
        force_mats = kwargs.pop('force_mats', False)
        force_constr = kwargs.pop('force_constr', False)
        force_om = kwargs.pop('force_om', False)
        self.init(force=force_init, force_mats=force_mats,
                  force_constr=force_constr, force_om=force_om)

        # --- solve optimization ---
        t0, _ = elapsed()
        _ = self.solve(**kwargs)
        status = self.om.prob.status
        self.exit_code = self.syms.status[status]
        self.converged = self.exit_code == 0
        _, s = elapsed(t0)
        self.exec_time = float(s.split(" ")[0])
        sstats = self.om.prob.solver_stats  # solver stats
        if sstats.num_iters is None:
            n_iter = -1
        else:
            n_iter = int(sstats.num_iters)
        n_iter_str = f"{n_iter} iterations " if n_iter > 1 else f"{n_iter} iteration "
        if self.exit_code == 0:
            msg = f"<{self.class_name}> solved as {status} in {s}, converged in "
            msg += n_iter_str + f"with {sstats.solver_name}."
            logger.warning(msg)
            self.unpack(res=None, **kwargs)
            self._post_solve()
            self.system.report()
            return True
        else:
            msg = f"{self.class_name} failed as {status} in "
            msg += n_iter_str + f"with {sstats.solver_name}!"
            logger.warning(msg)
            return False

    def load_json(self, path):
        """
        Load scheduling results from a json file.

        Parameters
        ----------
        path : str
            Path of the json file to load.

        Returns
        -------
        bool
            True if the loading is successful, False otherwise.

        .. versionadded:: 1.0.13
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON file: {e}")
            return False

        if not self.initialized:
            self.init()

        # Unpack variables and expressions from JSON
        for group, group_data in data.items():
            if not isinstance(group_data, dict):
                continue
            for key, values in group_data.items():
                if key == 'idx':
                    continue
                # Find the corresponding variable or expression
                if key in self.vars:
                    var = self.vars[key]
                    # Assign values to the variable
                    try:
                        var.v = np.array(values)
                    except Exception as e:
                        logger.warning(f"Failed to assign values to var '{key}': {e}")
                elif key in self.exprs:
                    continue
                elif key in self.exprcs:
                    exprc = self.exprcs[key]
                    # Assign values to the expression calculation
                    try:
                        exprc.v = np.array(values)
                    except Exception as e:
                        logger.warning(f"Failed to assign values to exprc '{key}': {e}")
        logger.info(f"Loaded results from {path}")
        return True

    def export_json(self, path=None):
        """
        Export scheduling results to a json file.

        Parameters
        ----------
        path : str, optional
            Path of the json file to export.

        Returns
        -------
        str
            The exported json file name

        .. versionadded:: 1.0.13
        """
        if not self.converged:
            logger.warning("Routine did not converge, aborting export.")
            return None

        path, file_name = get_export_path(self.system,
                                          self.class_name + '_out',
                                          path=path, fmt='json')

        data_dict = OrderedDict()

        # insert summary
        df = pd.DataFrame([summary_row])
        df.index.name = "uid"
        data_dict.update({summary_name: df.to_dict(orient='records')})

        # insert objective value
        data_dict.update(OrderedDict(Objective=self.obj.v))

        group_data(self, data_dict, self.vars, 'v')
        group_data(self, data_dict, self.exprs, 'v')
        group_data(self, data_dict, self.exprcs, 'v')

        with open(path, 'w') as f:
            json.dump(data_dict, f, indent=4,
                      default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        return file_name

    def export_csv(self, path=None):
        """
        Export scheduling results to a csv file.
        For multi-period routines, the column "Time" is the time index of
        ``timeslot.v``, which usually comes from ``EDTSlot`` or ``UCTSlot``.
        The rest columns are the variables registered in ``vars``.

        For single-period routines, the column "Time" have a pseduo value of "T1".

        Parameters
        ----------
        path : str, optional
            Path of the csv file to export.

        Returns
        -------
        str
            The exported csv file name
        """
        if not self.converged:
            logger.warning("Routine did not converge, aborting export.")
            return None

        path, file_name = get_export_path(self.system, self.class_name,
                                          path=path, fmt='csv')

        data_dict = initialize_data_dict(self)

        collect_data(self, data_dict, self.vars, 'v')
        collect_data(self, data_dict, self.exprs, 'v')
        collect_data(self, data_dict, self.exprcs, 'v')

        if 'T1' in data_dict['Time']:
            data_dict = OrderedDict([(k, [v]) for k, v in data_dict.items()])

        pd.DataFrame(data_dict).to_csv(path, index=False)

        return file_name

    def summary(self, **kwargs):
        """
        Summary interface
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.class_name} at {hex(id(self))}"

    def dc2ac(self, kloss=1.0, **kwargs):
        """
        Convert the DC-based results with ACOPF.
        """
        raise NotImplementedError

    def _check_attribute(self, key, value):
        """
        Check the attribute pair for valid names while instantiating the class.

        This function assigns `owner` to the model itself, assigns the name and tex_name.
        """
        if key in self.__dict__:
            existing_keys = []
            for rtn_type in ["constrs", "vars", "rparams", "services"]:
                if rtn_type in self.__dict__:
                    existing_keys += list(self.__dict__[rtn_type].keys())
            if key in existing_keys:
                msg = f"Attribute <{key}> already exists in <{self.class_name}>."
                logger.warning(msg)

        # register owner routine instance of following attributes
        if isinstance(value, (RBaseService)):
            value.rtn = self

    def __setattr__(self, key, value):
        """
        Overload the setattr function to register attributes.

        Parameters
        ----------
        key: str
            name of the attribute
        value:
            value of the attribute
        """

        # NOTE: value.id is not in use yet
        if isinstance(value, Var):
            value.id = len(self.vars)
        self._check_attribute(key, value)
        self._register_attribute(key, value)

        super(RoutineBase, self).__setattr__(key, value)

    def _register_attribute(self, key, value):
        """
        Register a pair of attributes to the routine instance.

        Called within ``__setattr__``, this is where the magic happens.
        Subclass attributes are automatically registered based on the variable type.
        """
        if isinstance(value, (Param, Var, Constraint, Objective, ExpressionCalc, Expression)):
            value.om = self.om
            value.rtn = self
        if isinstance(value, Param):
            self.params[key] = value
            self.om.params[key] = None  # cp.Parameter
        if isinstance(value, Var):
            self.vars[key] = value
            self.om.vars[key] = None  # cp.Variable
        elif isinstance(value, Constraint):
            self.constrs[key] = value
            self.om.constrs[key] = None  # cp.Constraint
        elif isinstance(value, Expression):
            self.exprs[key] = value
            self.om.exprs[key] = None  # cp.Expression
        elif isinstance(value, ExpressionCalc):
            self.exprcs[key] = value
        elif isinstance(value, RParam):
            self.rparams[key] = value
        elif isinstance(value, RBaseService):
            self.services[key] = value

    def update(self, params=None, build_mats=False):
        """
        Update the values of Parameters in the optimization model.

        This method is particularly important when some `RParams` are
        linked with system matrices.
        In such cases, setting `build_mats=True` is necessary to rebuild
        these matrices for the changes to take effect.
        This is common in scenarios involving topology changes, connection statuses,
        or load value modifications.
        If unsure, it is advisable to use `build_mats=True` as a precautionary measure.

        Parameters
        ----------
        params: Parameter, str, or list
            Parameter, Parameter name, or a list of parameter names to be updated.
            If None, all parameters will be updated.
        build_mats: bool
            True to rebuild the system matrices. Set to False to speed up the process
            if no system matrices are changed.
        """
        if not self.initialized:
            return self.init()
        t0, _ = elapsed()
        re_finalize = False
        # sanitize input
        sparams = []
        if params is None:
            sparams = [val for val in self.params.values()]
            build_mats = True
        elif isinstance(params, Param):
            sparams = [params]
        elif isinstance(params, str):
            sparams = [self.params[params]]
        elif isinstance(params, list):
            sparams = [self.params[param] for param in params if isinstance(param, str)]
            for param in sparams:
                param.update()

        for param in sparams:
            if param.optz is None:  # means no_parse=True
                re_finalize = True
                break

        self.system.mats.build(force=build_mats)

        if re_finalize:
            logger.warning(f"<{self.class_name}> reinit OModel due to non-parametric change.")
            self.om.evaluate(force=True)
            self.om.finalize(force=True)

        results = self.om.update(params=sparams)
        t0, s0 = elapsed(t0)
        logger.debug(f"Update params in {s0}.")
        return results

    def __delattr__(self, name):
        """
        Overload the delattr function to unregister attributes.

        Parameters
        ----------
        name: str
            name of the attribute
        """
        self._unregister_attribute(name)
        if name == "obj":
            self.obj = None
        else:
            super().__delattr__(name)  # Call the superclass implementation

    def _unregister_attribute(self, name):
        """
        Unregister a pair of attributes from the routine instance.

        Called within ``__delattr__``, this is where the magic happens.
        Subclass attributes are automatically unregistered based on the variable type.
        """
        if name in self.vars:
            del self.vars[name]
            if name in self.om.vars:
                del self.om.vars[name]
        elif name in self.rparams:
            del self.rparams[name]
        elif name in self.constrs:
            del self.constrs[name]
            if name in self.om.constrs:
                del self.om.constrs[name]
        elif name in self.services:
            del self.services[name]

    def enable(self, name):
        """
        Enable a constraint by name.

        Parameters
        ----------
        name: str or list
            name of the constraint to be enabled
        """
        if isinstance(name, list):
            constr_act = []
            for n in name:
                if n not in self.constrs:
                    logger.warning(f"Constraint <{n}> not found.")
                    continue
                if not self.constrs[n].is_disabled:
                    logger.warning(f"Constraint <{n}> has already been enabled.")
                    continue
                self.constrs[n].is_disabled = False
                self.om.finalized = False
                constr_act.append(n)
            if len(constr_act) > 0:
                msg = ", ".join(constr_act)
                logger.warning(f"Turn on constraints: {msg}")
            return True

        if name in self.constrs:
            if not self.constrs[name].is_disabled:
                logger.warning(f"Constraint <{name}> has already been enabled.")
            else:
                self.constrs[name].is_disabled = False
                self.om.finalized = False
                logger.warning(f"Turn on constraint <{name}>.")
            return True

    def disable(self, name):
        """
        Disable a constraint by name.

        Parameters
        ----------
        name: str or list
            name of the constraint to be disabled
        """
        if isinstance(name, list):
            constr_act = []
            for n in name:
                if n not in self.constrs:
                    logger.warning(f"Constraint <{n}> not found.")
                elif self.constrs[n].is_disabled:
                    logger.warning(f"Constraint <{n}> has already been disabled.")
                else:
                    self.constrs[n].is_disabled = True
                    self.om.finalized = False
                    constr_act.append(n)
            if len(constr_act) > 0:
                msg = ", ".join(constr_act)
                logger.warning(f"Turn off constraints: {msg}")
            return True

        if name in self.constrs:
            if self.constrs[name].is_disabled:
                logger.warning(f"Constraint <{name}> has already been disabled.")
            else:
                self.constrs[name].is_disabled = True
                self.om.finalized = False
                logger.warning(f"Turn off constraint <{name}>.")
            return True

        logger.warning(f"Constraint <{name}> not found.")

    def _post_add_check(self):
        """
        Post-addition check.
        """
        # --- reset routine status ---
        self.initialized = False
        self.exec_time = 0.0
        self.exit_code = 0
        # --- reset symprocessor status ---
        self._syms = False
        # --- reset optimization model status ---
        self.om.parsed = False
        self.om.evaluated = False
        self.om.finalized = False
        # --- reset OModel parser status ---
        self.om.parsed = False

    def addRParam(self,
                  name: str,
                  tex_name: Optional[str] = None,
                  info: Optional[str] = None,
                  src: Optional[str] = None,
                  unit: Optional[str] = None,
                  model: Optional[str] = None,
                  v: Optional[np.ndarray] = None,
                  indexer: Optional[str] = None,
                  imodel: Optional[str] = None,):
        """
        Add `RParam` to the routine.

        Parameters
        ----------
        name : str
            Name of this parameter. If not provided, `name` will be set
            to the attribute name.
        tex_name : str, optional
            LaTeX-formatted parameter name. If not provided, `tex_name`
            will be assigned the same as `name`.
        info : str, optional
            A description of this parameter
        src : str, optional
            Source name of the parameter.
        unit : str, optional
            Unit of the parameter.
        model : str, optional
            Name of the owner model or group.
        v : np.ndarray, optional
            External value of the parameter.
        indexer : str, optional
            Indexer of the parameter.
        imodel : str, optional
            Name of the owner model or group of the indexer.
        """
        item = RParam(name=name, tex_name=tex_name, info=info, src=src, unit=unit,
                      model=model, v=v, indexer=indexer, imodel=imodel)

        # add the parameter as an routine attribute
        setattr(self, name, item)

        # NOTE: manually register the owner of the parameter
        # This is skipped in ``addVars`` because of ``Var.__setattr__``
        item.rtn = self

        # check variable owner validity if given
        if model is not None:
            if item.model in self.system.groups.keys():
                item.is_group = True
                item.owner = self.system.groups[item.model]
            elif item.model in self.system.models.keys():
                item.owner = self.system.models[item.model]
            else:
                msg = f'Model indicator \'{item.model}\' of <{item.rtn.class_name}.{name}>'
                msg += ' is not a model or group. Likely a modeling error.'
                logger.warning(msg)

        self._post_add_check()
        return item

    def addService(self,
                   name: str,
                   value: np.ndarray,
                   tex_name: str = None,
                   unit: str = None,
                   info: str = None,
                   vtype: Type = None,):
        """
        Add `ValueService` to the routine.

        Parameters
        ----------
        name : str
            Instance name.
        value : np.ndarray
            Value.
        tex_name : str, optional
            TeX name.
        unit : str, optional
            Unit.
        info : str, optional
            Description.
        vtype : Type, optional
            Variable type.
        """
        item = ValueService(name=name, tex_name=tex_name,
                            unit=unit, info=info,
                            vtype=vtype, value=value)
        # add the service as an routine attribute
        setattr(self, name, item)

        self._post_add_check()

        return item

    def addConstrs(self,
                   name: str,
                   e_str: str,
                   info: Optional[str] = None,
                   is_eq: Optional[str] = False,):
        """
        Add `Constraint` to the routine. to the routine.

        Parameters
        ----------
        name : str
            Constraint name. One should typically assigning the name directly because
            it will be automatically assigned by the model. The value of ``name``
            will be the symbol name to be used in expressions.
        e_str : str
            Constraint expression string.
        info : str, optional
            Descriptive information
        is_eq : str, optional
            Flag indicating if the constraint is an equality constraint. False indicates
            an inequality constraint in the form of `<= 0`.
        """
        item = Constraint(name=name, e_str=e_str, info=info, is_eq=is_eq)
        # add the constraint as an routine attribute
        setattr(self, name, item)

        self._post_add_check()

        return item

    def addVars(self,
                name: str,
                model: Optional[str] = None,
                shape: Optional[Union[int, tuple]] = None,
                tex_name: Optional[str] = None,
                info: Optional[str] = None,
                src: Optional[str] = None,
                unit: Optional[str] = None,
                horizon: Optional[RParam] = None,
                nonneg: Optional[bool] = False,
                nonpos: Optional[bool] = False,
                cplx: Optional[bool] = False,
                imag: Optional[bool] = False,
                symmetric: Optional[bool] = False,
                diag: Optional[bool] = False,
                psd: Optional[bool] = False,
                nsd: Optional[bool] = False,
                hermitian: Optional[bool] = False,
                boolean: Optional[bool] = False,
                integer: Optional[bool] = False,
                pos: Optional[bool] = False,
                neg: Optional[bool] = False,):
        """
        Add a variable to the routine.

        Parameters
        ----------
        name : str, optional
            Variable name. One should typically assigning the name directly because
            it will be automatically assigned by the model. The value of ``name``
            will be the symbol name to be used in expressions.
        model : str, optional
            Name of the owner model or group.
        shape : int or tuple, optional
            Shape of the variable. If is None, the shape of `model` will be used.
        info : str, optional
            Descriptive information
        unit : str, optional
            Unit
        tex_name : str
            LaTeX-formatted variable symbol. If is None, the value of `name` will be
            used.
        src : str, optional
            Source variable name. If is None, the value of `name` will be used.
        lb : str, optional
            Lower bound
        ub : str, optional
            Upper bound
        horizon : ams.routines.RParam, optional
            Horizon idx.
        nonneg : bool, optional
            Non-negative variable
        nonpos : bool, optional
            Non-positive variable
        cplx : bool, optional
            Complex variable
        imag : bool, optional
            Imaginary variable
        symmetric : bool, optional
            Symmetric variable
        diag : bool, optional
            Diagonal variable
        psd : bool, optional
            Positive semi-definite variable
        nsd : bool, optional
            Negative semi-definite variable
        hermitian : bool, optional
            Hermitian variable
        bool : bool, optional
            Boolean variable
        integer : bool, optional
            Integer variable
        pos : bool, optional
            Positive variable
        neg : bool, optional
            Negative variable
        """
        if model is None and shape is None:
            raise ValueError("Either model or shape must be specified.")
        item = Var(name=name, tex_name=tex_name,
                   info=info, src=src, unit=unit,
                   model=model, shape=shape, horizon=horizon,
                   nonneg=nonneg, nonpos=nonpos,
                   cplx=cplx, imag=imag,
                   symmetric=symmetric, diag=diag,
                   psd=psd, nsd=nsd, hermitian=hermitian,
                   boolean=boolean, integer=integer,
                   pos=pos, neg=neg, )

        # add the variable as an routine attribute
        setattr(self, name, item)

        # check variable owner validity if given
        if model is not None:
            if item.model in self.system.groups.keys():
                item.is_group = True
                item.owner = self.system.groups[item.model]
            elif item.model in self.system.models.keys():
                item.owner = self.system.models[item.model]
            else:
                msg = (
                    f"Model indicator '{item.model}' of <{item.rtn.class_name}.{name}>"
                )
                msg += " is not a model or group. Likely a modeling error."
                logger.warning(msg)

        self._post_add_check()

        return item

    def _initial_guess(self):
        """
        Generate initial guess for the optimization model.
        """
        raise NotImplementedError


def initialize_data_dict(rtn: RoutineBase):
    """
    Initialize the data dictionary for export.

    Parameters
    ----------
    rtn : ams.routines.routine.RoutineBase
        The routine to collect data from

    Returns
    -------
    OrderedDict
        The initialized data dictionary.
    """
    if hasattr(rtn, 'timeslot'):
        timeslot = rtn.timeslot.v.copy()
        return OrderedDict([('Time', timeslot)])
    else:
        return OrderedDict([('Time', 'T1')])


def collect_data(rtn: RoutineBase, data_dict: Dict, items: Dict, attr: str):
    """
    Collect data for export.

    Parameters
    ----------
    rtn : ams.routines.routine.RoutineBase
        The routine to collect data from.
    data_dict : OrderedDict
        The data dictionary to populate.
    items : dict
        Dictionary of items to collect data from.
    attr : str
        Attribute to collect data for.
    """
    for key, item in items.items():
        if item.owner is None:
            continue
        idx_v = item.get_all_idxes()
        try:
            data_v = rtn.get(src=key, attr=attr, idx=idx_v,
                             horizon=rtn.timeslot.v if hasattr(rtn, 'timeslot') else None).round(6)
        except Exception as e:
            logger.debug(f"Error with collecting data for '{key}': {e}")
            data_v = [np.nan] * len(idx_v)
        data_dict.update(OrderedDict(zip([f'{key} {dev}' for dev in idx_v], data_v)))


def group_data(rtn: RoutineBase, data_dict: Dict, items: Dict, attr: str):
    """
    Collect data for export by groups, adding device idx in each group.
    This is useful when exporting to dictionary formats like JSON.

    Parameters
    ----------
    rtn : ams.routines.routine.RoutineBase
        The routine to collect data from.
    data_dict : Dict
        The data dictionary to populate.
    items : dict
        Dictionary of items to collect data from.
    attr : str
        Attribute to collect data for.

    .. versionadded:: 1.0.13
    """
    for key, item in items.items():
        if item.owner is None:
            continue
        if item.owner.class_name not in data_dict.keys():
            idx_v = item.get_all_idxes()
            data_dict[item.owner.class_name] = dict(idx=idx_v)
        else:
            idx_v = data_dict[item.owner.class_name]['idx']
        try:
            data_v = rtn.get(src=key, attr=attr, idx=idx_v,
                             horizon=rtn.timeslot.v if hasattr(rtn, 'timeslot') else None)
        except Exception as e:
            logger.warning(f"Error with collecting data for '{key}': {e}")
            data_v = [np.nan] * item.owner.n
        data_dict[item.owner.class_name][key] = data_v
