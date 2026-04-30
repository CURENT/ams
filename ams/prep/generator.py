"""
Source generator for AMS opt-layer codegen.

Walks a routine instance, rewrites each ``e_str`` into Python source
referencing a :class:`ams.core.routine_ns.RoutineNS` (``r.<name>``),
and produces a self-contained module of named callables.

Translation pipeline mirrors :class:`ams.core.symprocessor.SymProcessor`'s
``sub_map`` for the function-name part (``mul`` → ``cp.multiply``,
``dot`` → ``*``, etc.); the symbol-resolution part is replaced with
the simpler ``\\b<name>\\b → r.<name>`` so the generated code is
runtime-independent of OModel internals.

Errors are intentionally hard: a malformed ``e_str`` must surface at
prep time, not silently fall back to a slow regex+eval path.
"""

import hashlib
import inspect
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cvxpy as cp

import ams


_FUNCTION_REWRITES = OrderedDict([
    (r'\b(\w+)\s+dot\s+(\w+)\b', r'\1 * \2'),
    (r' dot ', r' * '),
    (r'\bsum\b', 'cp.sum'),
    (r'\bvar\b', 'cp.Variable'),
    (r'\bparam\b', 'cp.Parameter'),
    (r'\bconst\b', 'cp.Constant'),
    (r'\bproblem\b', 'cp.Problem'),
    (r'\bmultiply\b', 'cp.multiply'),
    (r'\bmul\b', 'cp.multiply'),
    (r'\bvstack\b', 'cp.vstack'),
    (r'\bnorm\b', 'cp.norm'),
    (r'\bpos\b', 'cp.pos'),
    (r'\bpower\b', 'cp.power'),
    (r'\bsign\b', 'cp.sign'),
    (r'\bmaximum\b', 'cp.maximum'),
    (r'\bminimum\b', 'cp.minimum'),
    (r'\bsquare\b', 'cp.square'),
    (r'\bquad_over_lin\b', 'cp.quad_over_lin'),
    (r'\bdiag\b', 'cp.diag'),
    (r'\bquad_form\b', 'cp.quad_form'),
    (r'\bsum_squares\b', 'cp.sum_squares'),
])

# Bare identifiers above that, if used as a routine symbol name, would be
# silently consumed by the function-rewrite pass before symbol resolution
# runs. ``dot`` is also reserved (the two top-of-list patterns rewrite it
# to ``*``). Hard-fail at codegen rather than emit subtly broken code.
_RESERVED_SYMBOL_NAMES = frozenset({
    'sum', 'var', 'param', 'const', 'problem', 'multiply', 'mul',
    'vstack', 'norm', 'pos', 'power', 'sign', 'maximum', 'minimum',
    'square', 'quad_over_lin', 'diag', 'quad_form', 'sum_squares', 'dot',
})


# Static tex_map templates — mirror those built in
# :class:`ams.core.symprocessor.SymProcessor.__init__`. The codegen
# needs the same templates plus the per-symbol ``\b<name>\b → tex_name``
# rules (added in :func:`_build_tex_map`).
_TEX_TEMPLATES = OrderedDict([
    (r'\*\*(\d+)', '^{\\1}'),
    (r'\b(\w+)\s*\*\s*(\w+)\b', r'\1 \2'),
    (r'\@', r' '),
    (r'dot', r' '),
    (r'sum_squares\((.*?)\)', r"SUM((\1))^2"),
    (r'multiply\(([^,]+), ([^)]+)\)', r'\1 \2'),
    (r'\bnp.linalg.pinv(\d+)', r'\1^{\-1}'),
    (r'\bpos\b', 'F^{+}'),
    (r'mul\((.*?),\s*(.*?)\)', r'\1 \2'),
    (r'\bmul\b\((.*?),\s*(.*?)\)', r'\1 \2'),
    (r'\bsum\b', 'SUM'),
    (r'power\((.*?),\s*(\d+)\)', r'\1^\2'),
    (r'(\w+).dual_variables\[0\]', r'\phi[\1]'),
])


def _build_tex_map(routine) -> "OrderedDict[str, str]":
    """Build the tex_map for codegen-time LaTeX rendering.

    Mirrors what :meth:`SymProcessor.generate_symbols` would put in
    ``rtn.syms.tex_map`` — the static templates plus a
    ``\\b<name>\\b → <tex_name>`` rule per symbol. The codegen running
    this against an item's ``e_str`` produces the same string the
    legacy runtime path produced.
    """
    tex_map = OrderedDict(_TEX_TEMPLATES)
    for name, var in routine.vars.items():
        tex_map[rf'\b{name}\b'] = var.tex_name or name
    for name, rparam in routine.rparams.items():
        tex_map[rf'\b{name}\b'] = rparam.tex_name or name
    for name, service in routine.services.items():
        tex_map[rf'\b{name}\b'] = service.tex_name or name
    for name, expr in routine.exprs.items():
        tex_map[rf'\b{name}\b'] = expr.tex_name or name
    cfg_tex = getattr(routine.config, 'tex_names', {}) or {}
    for key in routine.config.as_dict():
        tex_map[rf'\b{key}\b'] = cfg_tex.get(key, key)
    return tex_map


def _collect_symbol_names(routine) -> list:
    """All symbol names the runtime :class:`RoutineNS` resolves."""
    names = set()
    for category in (routine.vars, routine.rparams, routine.services,
                     routine.exprs, routine.constrs):
        names.update(category.keys())
    names.update(routine.config.as_dict().keys())
    names.add('sys_f')
    names.add('sys_mva')
    collisions = names & _RESERVED_SYMBOL_NAMES
    if collisions:
        raise ValueError(
            f"codegen: routine <{type(routine).__name__}> has symbol(s) "
            f"{sorted(collisions)} that collide with the function-rewrite "
            f"vocabulary ({sorted(_RESERVED_SYMBOL_NAMES)}). Rename the "
            f"symbol(s) — they would be silently consumed before symbol "
            f"resolution runs."
        )
    return sorted(names)


def _build_symbol_regex(names: list):
    """Single alternation regex matching any symbol name (with word boundaries).

    Sort by length descending so a longer name takes precedence over a
    shorter prefix when both are in the alternation (e.g. ``pmax`` over
    ``p`` if both existed).
    """
    if not names:
        return None
    sorted_names = sorted(names, key=lambda n: (-len(n), n))
    return re.compile(r'\b(' + '|'.join(re.escape(n) for n in sorted_names) + r')\b')


def _rewrite(e_str: str, function_rewrites, symbol_regex) -> str:
    """Apply the codegen rewrites in two passes.

    Pass 1: function-name rewrites (``mul → cp.multiply``, ``dot → *``,
    ``sum → cp.sum``, etc.). These can run sequentially because their
    outputs (``cp.multiply``, ``*``) never match a later input pattern.

    Pass 2: a *single* substitution that resolves every symbol name at
    once, ``r.<name>``. Sequential per-name substitutions cascade —
    e.g. when a routine has a symbol named ``r`` (DOPF.r = line
    resistance), substituting ``\\br\\b → r.r`` first, then later
    seeing the already-emitted ``r.Bf`` and matching ``r`` again yields
    ``r.r.Bf``. Single-pass alternation prevents that.
    """
    code = e_str
    for pattern, replacement in function_rewrites.items():
        try:
            code = re.sub(pattern, replacement, code)
        except re.error as exc:
            raise ValueError(
                f"codegen: regex failure on pattern {pattern!r} "
                f"applying to {e_str!r}: {exc}"
            )
    if symbol_regex is not None:
        code = symbol_regex.sub(r'r.\1', code)
    return code


def _validate_python(source: str, where: str) -> None:
    """Compile-check a snippet so syntax errors surface at prep time, not later."""
    try:
        compile(source, where, 'exec')
    except SyntaxError as exc:
        raise SyntaxError(
            f"codegen produced invalid Python at {where}:\n"
            f"  source: {source!r}\n"
            f"  error:  {exc}"
        ) from exc


def source_md5(routine_cls) -> str:
    """md5 of the routine class's source file.

    Used as the cache key for the generated pycode. A change to the
    routine module (new e_str, edited mixin) invalidates pycode and
    triggers a regen on the next instantiation.

    NOTE: does not currently traverse the MRO across multiple files.
    If a parent class lives in a different module and that file
    changes without the leaf module changing, the cache won't notice.
    Acceptable for now (`ams prep --force` is the escape hatch).
    """
    src_file = inspect.getsourcefile(routine_cls)
    if src_file is None:
        return 'no-source'
    # md5 is fine here — used as a cache key for code regeneration, not
    # for any security or integrity property. nosec for Codacy's Bandit
    # engine (which ignores [tool.bandit] skips in pyproject.toml).
    return hashlib.md5(Path(src_file).read_bytes()).hexdigest()  # nosec B324


def _emit_callable(prefix: str, name: str, body: str) -> List[str]:
    """Render one ``def _<prefix>_<name>(r): return <body>``."""
    snippet = f'def _{prefix}_{name}(r):\n    return {body}\n'
    _validate_python(snippet, where=f'<codegen:{prefix}_{name}>')
    return [snippet]


def _render_tex(item, tex_map) -> str:
    """Run an item's ``e_str`` through the same tex_map pipeline the
    documenter uses (``ams.core.documenter._tex_pre``).

    Calling the existing helper preserves the map_before/map_post
    sigil-escape dance so backslashed replacements (``\\sum``,
    ``\\theta``, ``\\eta`` …) survive ``re.sub``'s replacement-string
    interpretation. The result is identical to the runtime
    documenter path; we just compute it once at codegen time and
    embed in pycode for items that won't have ``e_str`` post-init.
    """
    from ams.core.documenter import _tex_pre

    class _StubDocm:
        class parent:
            class_name = '<codegen>'

    return _tex_pre(_StubDocm(), item, tex_map)


def _emit_tex_string(prefix: str, name: str, tex: str) -> List[str]:
    """Render ``_<prefix>_<name>_tex = <repr>`` — a pre-rendered LaTeX string."""
    return [f'_{prefix}_{name}_tex = {tex!r}\n']


def generate_for_routine(routine, *, header_extra: str = '') -> str:
    """Generate pycode source for one routine instance.

    Parameters
    ----------
    routine : ams.routines.routine.RoutineBase
        A fully-constructed routine — its ``vars`` / ``rparams`` /
        ``constrs`` / ``exprs`` / ``exprcs`` / ``obj`` registries must
        all be populated (i.e. after ``__init__`` has run).
    header_extra : str, optional
        Extra lines to splice into the module header (e.g.
        provenance for tests).

    Returns
    -------
    source : str
        Python source for the generated module. Caller writes it to
        ``~/.ams/pycode/<class_name_lower>.py``.

    Raises
    ------
    SyntaxError
        If any rewritten expression is not valid Python.
    ValueError
        On regex failure.
    """
    function_rewrites = _FUNCTION_REWRITES
    symbol_regex = _build_symbol_regex(_collect_symbol_names(routine))
    tex_map = _build_tex_map(routine)
    cls = type(routine)
    src_file = inspect.getsourcefile(cls)

    parts: List[str] = []
    parts.append(f'"""Generated pycode for {cls.__name__}.\n')
    parts.append('Do not edit; run `ams prep` to regenerate.\n')
    if src_file is not None:
        parts.append(f'Auto-generated from {Path(src_file).name} '
                     f'at {datetime.now().isoformat(timespec="seconds")}.\n')
    if header_extra:
        parts.append(header_extra + '\n')
    parts.append('"""\n\n')

    parts.append('import cvxpy as cp  # noqa: F401\n')
    parts.append('import numpy as np  # noqa: F401\n\n')

    parts.append(f'class_name = {cls.__name__!r}\n')
    parts.append(f'md5 = {source_md5(cls)!r}\n')
    parts.append(f'ams_version = {ams.__version__!r}\n')
    parts.append(f'cvxpy_version = {cp.__version__!r}\n')
    # ``pristine`` asserts the cache was generated from a routine instance
    # untouched by user customizations (a fresh ``ams.System`` constructed
    # inside ``_link_pycode``, not the user's ``sp.DCOPF`` which may carry
    # ``addConstrs`` / ``e_str+=`` mutations). ``_link_pycode`` rejects any
    # cache without this marker as stale, which auto-heals polluted caches
    # written by older AMS versions that codegen'd against the live
    # instance.
    parts.append('pristine = True\n\n')

    # --- Expressions ---
    for name, expr in routine.exprs.items():
        if expr.e_fn is not None or expr.e_str is None:
            continue
        body = _rewrite(expr.e_str, function_rewrites, symbol_regex)
        parts.extend(_emit_callable('expr', name, body))
        parts.extend(_emit_tex_string('expr', name, _render_tex(expr, tex_map)))
        parts.append('\n')

    # --- Constraints ---
    # Codegen convention: emit the LHS expression only. Constraint.evaluate
    # applies ``<= 0`` / ``== 0`` based on ``is_eq``. This lets ``.e`` recover
    # a numpy LHS during a failed/incomplete solve via ``NumericRoutineNS``;
    # the legacy ``return <body> <= 0`` form short-circuits to a numpy bool
    # when every operand is numpy, which loses the LHS.
    for name, constr in routine.constrs.items():
        if constr.e_fn is not None or constr.e_str is None:
            continue
        body = _rewrite(constr.e_str, function_rewrites, symbol_regex)
        parts.extend(_emit_callable('constr', name, body))
        parts.extend(_emit_tex_string('constr', name, _render_tex(constr, tex_map)))
        parts.append('\n')

    # --- ExpressionCalcs ---
    for name, exprc in routine.exprcs.items():
        if exprc.e_str is None:
            continue
        body = _rewrite(exprc.e_str, function_rewrites, symbol_regex)
        parts.extend(_emit_callable('exprcalc', name, body))
        parts.extend(_emit_tex_string('exprcalc', name, _render_tex(exprc, tex_map)))
        parts.append('\n')

    # --- Objective ---
    if routine.obj is not None and routine.obj.e_fn is None and routine.obj.e_str is not None:
        body = _rewrite(routine.obj.e_str, function_rewrites, symbol_regex)
        parts.extend(_emit_callable('obj', routine.obj.name, body))
        parts.extend(_emit_tex_string('obj', routine.obj.name, _render_tex(routine.obj, tex_map)))
        parts.append('\n')

    return ''.join(parts)


def write_for_routine(routine, target: Optional[Path] = None) -> Path:
    """Write the generated pycode for ``routine`` to disk.

    Default target: ``~/.ams/pycode/<class_name_lower>.py``.
    """
    if target is None:
        target = Path.home() / '.ams' / 'pycode' / f'{type(routine).__name__.lower()}.py'
    target.parent.mkdir(parents=True, exist_ok=True)
    src = generate_for_routine(routine)
    target.write_text(src)
    return target
