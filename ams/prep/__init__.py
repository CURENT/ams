"""
AOT codegen for AMS routines (Phase 4.5+).

Walks a fully-constructed routine instance and emits a Python module
of named callables — one per ``Constraint``/``Expression``/``Objective``/
``ExpressionCalc`` with an ``e_str``. The generated functions take a
:class:`ams.core.routine_ns.RoutineNS` and return the same shape
``e_fn=...`` already accepts, so the routine can wire them in at init
time without any regex/eval at runtime.

Authors keep writing ``e_str`` strings — the terse DSL is preserved.
``ams prep`` (or auto-prep on first init) is what produces the runtime
artefacts.

Public API:

- :func:`generate_for_routine` — emit source for one routine.
- :func:`prep_all` — emit source for every registered routine class.
"""

import logging
import shutil
from pathlib import Path
from typing import Iterable, Optional

import cvxpy as _cp

from ams.prep.generator import generate_for_routine, source_md5, write_for_routine  # noqa: F401

logger = logging.getLogger(__name__)


def pycode_dir() -> Path:
    """The on-disk directory holding generated pycode."""
    return Path.home() / '.ams' / 'pycode'


_PRISTINE_SYSTEM = None


def _get_pristine_system():
    """
    Lazily construct (and cache) an ``ams.System`` to use as the source
    of pristine routine instances for codegen.

    Per-process singleton so ``_link_pycode`` can drive codegen against
    untouched routines no matter how mutated the user's working
    ``System`` has become. Cheap to keep around — a no-input System is
    a few MB.
    """
    global _PRISTINE_SYSTEM
    if _PRISTINE_SYSTEM is None:
        import ams as _ams
        _PRISTINE_SYSTEM = _ams.System(no_input=True, default_config=True)
    return _PRISTINE_SYSTEM


def clean(verbose: bool = True) -> None:
    """Remove the entire pycode cache directory."""
    target = pycode_dir()
    if target.exists():
        shutil.rmtree(target)
        if verbose:
            logger.info(f"Removed {target}")
    elif verbose:
        logger.info(f"Nothing to clean: {target} does not exist")


def prep_all(routines: Optional[Iterable[str]] = None,
             force: bool = False,
             verbose: bool = True) -> int:
    """
    Generate (or refresh) pycode for every registered routine.

    Parameters
    ----------
    routines : iterable of str, optional
        Subset of routine class names to prep. If omitted, all routines
        registered in :data:`ams.routines.all_routines` are prepared.
    force : bool, optional
        If True, regenerate even when the cached md5 matches the current
        source.
    verbose : bool, optional
        If True (default), log each routine prepared.

    Returns
    -------
    n : int
        Number of routine pycode files written.
    """
    from ams.routines import all_routines

    # Build the set of routine class names to prep.
    wanted = None
    if routines is not None:
        wanted = {r.lower() for r in routines}

    # Reuse the singleton pristine System so we don't construct two if
    # ``_link_pycode`` already built one for an in-process auto-prep.
    system = _get_pristine_system()
    n = 0
    for group, names in all_routines.items():
        for name in names:
            if wanted is not None and name.lower() not in wanted:
                continue
            rtn = getattr(system, name, None)
            if rtn is None:
                logger.debug(f"Skip {name}: not present on System")
                continue
            target = pycode_dir() / f'{name.lower()}.py'
            if not force and target.exists():
                # Mirror the staleness check in _link_pycode: both md5 AND
                # cvxpy_version must match, otherwise regen.
                expected = source_md5(type(rtn))
                try:
                    txt = target.read_text()
                    md5_ok = f"md5 = {expected!r}" in txt
                    cvx_ok = f"cvxpy_version = {_cp.__version__!r}" in txt
                    pristine_ok = "pristine = True" in txt
                    if md5_ok and cvx_ok and pristine_ok:
                        if verbose:
                            logger.debug(f"  {name}: up-to-date")
                        continue
                except OSError:
                    pass
            write_for_routine(rtn, target)
            if verbose:
                logger.info(f"  prepped {name} -> {target}")
            n += 1
    return n
