"""
Source generator for AMS opt-layer codegen.

Walks a routine instance, rewrites each ``e_str`` into Python source
referencing a :class:`ams.core.routine_ns.RoutineNS` (``r.<name>``),
and produces a self-contained module of named callables.

Routines author canonical CVXPY (``cp.sum(...)``, ``cp.multiply(...)``)
directly. Codegen does only symbol resolution
(``\\b<name>\\b → r.<name>``); there is no operator-name rewrite stage.
The negative lookbehind on the symbol regex skips the ``name`` part of
``cp.name`` attribute access, so ``cp.sum(r.pg)`` survives the pass even
if a routine declares a symbol called ``sum``.

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


# Bumped whenever the shape of generated pycode changes (new header
# field, signature change on emitted callables, etc.). ``_link_pycode``
# rejects any cache whose ``pycode_format_version`` doesn't match this
# value, forcing a regen on the next run. Older caches that predate the
# field (no attribute) are likewise rejected. Auto-heal pattern: same
# as the ``pristine`` marker added in PR #242.
PYCODE_FORMAT_VERSION = 3


# Routine symbol names that would silently shadow a CVXPY atom in the
# eval-fallback rewrite path. The codegen path is safe (the ``(?<!\.)``
# lookbehind protects ``cp.<name>`` attribute access), but
# :func:`ams.opt._runtime_eval._resolve` rewrites bare names via
# ``_build_symbol_regex`` against the routine's symbol registries — so
# a routine declaring a symbol named ``sum`` plus a user appending a
# bare ``sum(pg)`` to ``obj.e_str`` would yield ``r.sum(r.pg)``, which
# is the user's symbol, not ``cp.sum``. Reject the collision at
# codegen / generate_symbols time so the trap surfaces at routine
# definition, not as silent nonsense at solve time.
#
# Static fallback: the closed set we know mattered when the guard
# shipped (PR #244-#248, cvxpy-namespace passthrough). The runtime
# set below is unioned with this — never narrower — so a CVXPY API
# shape change (atoms moved/removed) can shrink derivation but
# can't shrink the guarded set.
_STATIC_RESERVED_CVXPY_ATOM_NAMES = frozenset({
    'sum', 'multiply', 'vstack', 'hstack', 'power', 'norm', 'pos', 'neg',
    'square', 'quad_form', 'sum_squares', 'diag', 'maximum', 'minimum',
    'abs', 'exp', 'log', 'sqrt', 'inv_pos',
})


def _derive_reserved_cvxpy_atom_names():
    """Discover CVXPY atom-callable names exposed at the ``cp.`` top.

    Walks ``cvxpy.atoms`` (the atoms package) and keeps each public
    name that's also exposed as a callable on the top-level ``cvxpy``
    namespace. The result tracks whatever atoms the installed CVXPY
    version ships, so an atom added in a future release is guarded
    automatically (the static fallback was a snapshot — this isn't).
    Always unioned with :data:`_STATIC_RESERVED_CVXPY_ATOM_NAMES`,
    so it's a strict superset.
    """
    derived = set()
    try:
        atoms_mod = cp.atoms
    except AttributeError:
        return _STATIC_RESERVED_CVXPY_ATOM_NAMES
    for name in dir(atoms_mod):
        if name.startswith('_'):
            continue
        attr = getattr(cp, name, None)
        if attr is not None and callable(attr):
            derived.add(name)
    return frozenset(derived | _STATIC_RESERVED_CVXPY_ATOM_NAMES)


RESERVED_CVXPY_ATOM_NAMES = _derive_reserved_cvxpy_atom_names()


def _check_reserved_collisions(routine, names) -> None:
    """Raise if any routine symbol collides with a CVXPY atom name."""
    bad = sorted(set(names) & RESERVED_CVXPY_ATOM_NAMES)
    if bad:
        raise ValueError(
            f"Routine <{routine.class_name}> declares symbol(s) "
            f"{bad} that collide with CVXPY atom names. The "
            "eval-fallback helper (ams.opt._runtime_eval) would "
            "silently rewrite a bare ``<name>(...)`` in a user "
            "customization to ``r.<name>(...)``, shadowing "
            "``cp.<name>``. Rename the routine symbol(s) — see "
            "RESERVED_CVXPY_ATOM_NAMES in ams/prep/generator.py."
        )


# Static tex_map templates — mirror those built in
# :class:`ams.core.symprocessor.SymProcessor.__init__`. The codegen
# needs the same templates plus the per-symbol ``\b<name>\b → tex_name``
# rules (added in :func:`_build_tex_map`).
#
# The first rule strips the ``cp.`` Python-module prefix from
# canonical-CVXPY e_str (added in the namespace-passthrough migration).
# Without it, ``cp.sum(cp.multiply(...))`` would render as
# ``cp.\sum(cp.c_{...} ...)`` — the ``cp.`` is Python plumbing, never
# math. Mirrors the same first-rule in
# :attr:`SymProcessor.tex_map`; both must agree.
_TEX_TEMPLATES = OrderedDict([
    # ``\*\*`` MUST come before the ``cp.`` stripper. ``_tex_pre`` runs
    # ``expr.replace('*', ' ')`` after every substitution, so any rule that
    # leaves a ``**`` unconverted will see it shredded into two spaces.
    (r'\*\*(\d+)', '^{\\1}'),
    (r'\bcp\.(\w+)', r'\1'),
    (r'\b(\w+)\s*\*\s*(\w+)\b', r'\1 \2'),
    (r'\@', r' '),
    (r'dot', r' '),
    (r'sum_squares\((.*?)\)', r"SUM((\1))^2"),
    # ``multiply\(...\)`` runs twice — the first pass only flattens
    # non-overlapping matches, so a nested ``multiply(multiply(a, b), c)``
    # leaves the outer call intact after one pass. The second pass uses
    # the ``\b`` form so its dict key differs (OrderedDict dedupes
    # identical keys); behavior is identical. Mirrors the two-rule
    # ``mul``/``\bmul\b`` chain below.
    (r'multiply\(([^,]+), ([^)]+)\)', r'\1 \2'),
    (r'\bmultiply\b\(([^,]+), ([^)]+)\)', r'\1 \2'),
    (r'\bnp.linalg.pinv(\d+)', r'\1^{\-1}'),
    (r'\bpos\b', 'F^{+}'),
    (r'mul\((.*?),\s*(.*?)\)', r'\1 \2'),
    (r'\bmul\b\((.*?),\s*(.*?)\)', r'\1 \2'),
    (r'\bsum\b', 'SUM'),
    (r'power\((.*?),\s*(\d+)\)', r'\1^\2'),
    (r'(\w+).dual_variables\[0\]', r'\phi[\1]'),
    # Relational operators must come AFTER the `\bcp.X` stripper above
    # but ordering within this trio doesn't matter (no overlap among
    # `<=`, `>=`, `==`). `==` -> `=` because `\eq` would be ambiguous
    # under math-mode rendering. Replacements use double-backslash so
    # `re.sub` emits a literal `\leq` / `\geq` (single-backslash forms
    # raise re.PatternError on Python 3.12+).
    (r'<=', r'\\leq'),
    (r'>=', r'\\geq'),
    (r'==', '='),
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
    sorted_names = sorted(names)
    _check_reserved_collisions(routine, sorted_names)
    return sorted_names


def _build_symbol_regex(names: list):
    """Single alternation regex matching any symbol name (with word boundaries).

    Sort by length descending so a longer name takes precedence over a
    shorter prefix when both are in the alternation (e.g. ``pmax`` over
    ``p`` if both existed).

    The ``(?<!\\.)`` lookbehind skips identifiers preceded by a dot —
    so ``cp.sum(r.pg)`` is left alone even when ``sum`` happens to be a
    routine symbol name. This is what makes the function-rewrite layer
    safe to delete: routines are now expected to author canonical
    ``cp.X(...)`` directly, and the symbol-resolution pass must not
    touch the ``X`` part of attribute access.
    """
    if not names:
        return None
    sorted_names = sorted(names, key=lambda n: (-len(n), n))
    return re.compile(
        r'(?<!\.)\b(' + '|'.join(re.escape(n) for n in sorted_names) + r')\b'
    )


def _rewrite(e_str: str, symbol_regex) -> str:
    """Resolve symbol names to ``r.<name>`` in one pass.

    A *single* alternation substitution covers every symbol; sequential
    per-name substitutions would cascade — e.g. when a routine has a
    symbol named ``r`` (DOPF.r = line resistance), substituting
    ``\\br\\b → r.r`` first and then seeing the already-emitted
    ``r.Bf`` would match ``r`` again and yield ``r.r.Bf``. Single-pass
    alternation prevents that, and the regex's ``(?<!\\.)`` lookbehind
    leaves attribute access (``cp.sum``, ``r.pg``) untouched.
    """
    if symbol_regex is None:
        return e_str
    return symbol_regex.sub(r'r.\1', e_str)


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
    """Cache-key hash of the routine class's source file.

    Used as the cache key for the generated pycode. A change to the
    routine module (new e_str, edited mixin) invalidates pycode and
    triggers a regen on the next instantiation. The function name still
    reads ``_md5`` for historical reasons; the underlying digest is
    sha256 (no cryptographic intent, just a cache key — sha256 makes
    Codacy's Semgrep crypto rule happy without complicating the call
    site with ``# nosec`` markers).

    NOTE: does not currently traverse the MRO across multiple files.
    If a parent class lives in a different module and that file
    changes without the leaf module changing, the cache won't notice.
    Acceptable for now (`ams prep --force` is the escape hatch).
    """
    src_file = inspect.getsourcefile(routine_cls)
    if src_file is None:
        return 'no-source'
    return hashlib.sha256(Path(src_file).read_bytes()).hexdigest()


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
    parts.append(f'pycode_format_version = {PYCODE_FORMAT_VERSION!r}\n')
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
        body = _rewrite(expr.e_str, symbol_regex)
        parts.extend(_emit_callable('expr', name, body))
        parts.extend(_emit_tex_string('expr', name, _render_tex(expr, tex_map)))
        parts.append('\n')

    # --- Constraints ---
    # Codegen emits the rewritten ``e_str`` verbatim — relational operator
    # included (post `is_eq` retirement). The wired callable returns a
    # ``cp.constraints.Constraint``; ``Constraint.evaluate`` stores it
    # directly. ``.e`` recovers the numpy LHS via ``optz._expr.value``,
    # which CVXPY canonicalizes to ``lhs - rhs`` for any inequality
    # direction — same diagnostic semantics as the prior LHS-only emit.
    for name, constr in routine.constrs.items():
        if constr.e_fn is not None or constr.e_str is None:
            continue
        body = _rewrite(constr.e_str, symbol_regex)
        parts.extend(_emit_callable('constr', name, body))
        parts.extend(_emit_tex_string('constr', name, _render_tex(constr, tex_map)))
        parts.append('\n')

    # --- ExpressionCalcs ---
    for name, exprc in routine.exprcs.items():
        if exprc.e_str is None:
            continue
        body = _rewrite(exprc.e_str, symbol_regex)
        parts.extend(_emit_callable('exprcalc', name, body))
        parts.extend(_emit_tex_string('exprcalc', name, _render_tex(exprc, tex_map)))
        parts.append('\n')

    # --- Objective ---
    if routine.obj is not None and routine.obj.e_fn is None and routine.obj.e_str is not None:
        body = _rewrite(routine.obj.e_str, symbol_regex)
        parts.extend(_emit_callable('obj', routine.obj.name, body))
        parts.extend(_emit_tex_string('obj', routine.obj.name, _render_tex(routine.obj, tex_map)))
        parts.append('\n')

    return ''.join(parts)


def write_for_routine(routine, target: Optional[Path] = None) -> Path:
    """Write the generated pycode for ``routine`` to disk.

    Default target: ``~/.ams/pycode/<class_name_lower>.py``.
    """
    if target is None:
        # Imported here (not at module top) to avoid an import cycle:
        # ``ams.prep.__init__`` already imports from this module.
        from ams.prep import pycode_dir
        target = pycode_dir() / f'{type(routine).__name__.lower()}.py'
    target.parent.mkdir(parents=True, exist_ok=True)
    src = generate_for_routine(routine)
    target.write_text(src)
    return target
