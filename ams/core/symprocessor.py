"""
Symbolic processor class for AMS routines,
revised from `andes.core.symprocessor`.

Revised from the ANDES project:
https://github.com/CURENT/andes

Original author: Hantao Cui

License: GNU General Public License v3.0 (GPL-3.0)
"""

import logging
from collections import OrderedDict

import sympy as sp

from ams.utils.misc import elapsed

logger = logging.getLogger(__name__)


class SymProcessor:
    """
    Class for symbolic processing in AMS routine.

    Parameters
    ----------
    parent : ams.routines.base.BaseRoutine
        Routine instance to process.

    Attributes
    ----------
    tex_map : dict
        Tex substitution map for documentation.
    """

    def __init__(self, parent):
        self.parent = parent
        self.inputs_dict = OrderedDict()
        self.services_dict = OrderedDict()
        self.config = parent.config
        self.class_name = parent.class_name
        self.tex_names = OrderedDict()

        # First rule strips the ``cp.`` Python-module prefix from
        # canonical-CVXPY e_str (added in the namespace-passthrough
        # migration). Without it, ``cp.sum(cp.multiply(...))`` would
        # render as ``cp.\sum(cp.c_{...} ...)`` — the ``cp.`` is Python
        # plumbing, never math. Mirrors
        # :data:`ams.prep.generator._TEX_TEMPLATES`; both must agree.
        # ``\*\*`` MUST come before the ``cp.`` stripper —
        # ``_tex_pre`` runs ``expr.replace('*', ' ')`` after every
        # substitution, shredding any unconverted ``**`` into two
        # spaces.
        self.tex_map = OrderedDict([
            (r'\*\*(\d+)', '^{\\1}'),
            (r'\bcp\.(\w+)', r'\1'),
            (r'\b(\w+)\s*\*\s*(\w+)\b', r'\1 \2'),
            (r'\@', r' '),
            (r'dot', r' '),
            (r'sum_squares\((.*?)\)', r"SUM((\1))^2"),
            # ``multiply\(...\)`` runs twice (second key is the
            # ``\b`` form to bypass OrderedDict key dedup) to catch
            # nested ``multiply(multiply(...))``. Same trick as the
            # ``mul``/``\bmul\b`` pair below; see
            # :data:`ams.prep.generator._TEX_TEMPLATES`.
            (r'multiply\(([^,]+), ([^)]+)\)', r'\1 \2'),
            (r'\bmultiply\b\(([^,]+), ([^)]+)\)', r'\1 \2'),
            (r'\bnp.linalg.pinv(\d+)', r'\1^{\-1}'),
            (r'\bpos\b', 'F^{+}'),
            (r'mul\((.*?),\s*(.*?)\)', r'\1 \2'),
            (r'\bmul\b\((.*?),\s*(.*?)\)', r'\1 \2'),
            (r'\bsum\b', 'SUM'),
            (r'power\((.*?),\s*(\d+)\)', r'\1^\2'),
            (r'(\w+).dual_variables\[0\]', r'\phi[\1]'),
        ])

        self.status = {
            'optimal': 0,
            'infeasible': 1,
            'unbounded': 2,
            'infeasible_inaccurate': 3,
            'unbounded_inaccurate': 4,
            'optimal_inaccurate': 5,
            'solver_error': 6,
            'time_limit': 7,
            'interrupted': 8,
            'unknown': 9,
            'infeasible_or_unbounded': 1.5,
            'user_limit': 10,
        }

    def generate_symbols(self, force_generate=False):
        """
        Generate symbols for all variables.
        """
        logger.debug(f'Entering symbol generation for <{self.parent.class_name}>')

        if (not force_generate) and self.parent._syms:
            logger.debug(' - Symbols already generated')
            return True
        t, _ = elapsed()

        # Invalidate the eval-helper symbol regex cache; the routine's
        # symbol set may have changed since the last run (addRParam,
        # addConstrs, etc.). See ams.opt._runtime_eval._get_symbol_regex.
        self._eval_symbol_regex = None

        # Reject routine symbol names that collide with CVXPY atoms;
        # see RESERVED_CVXPY_ATOM_NAMES in ams.prep.generator. The
        # codegen path runs the same check via _collect_symbol_names,
        # but routines can hit the eval-fallback helper without ever
        # going through codegen (e.g. addConstrs at runtime), so this
        # side enforces the contract independently.
        from ams.prep.generator import _check_reserved_collisions
        _sym_names = set()
        for _category in (self.parent.vars, self.parent.rparams,
                          self.parent.services, self.parent.exprs,
                          self.parent.constrs):
            _sym_names.update(_category.keys())
        _check_reserved_collisions(self.parent, _sym_names)

        # process tex_names defined in routines
        # -----------------------------------------------------------
        for key in self.parent.tex_names.keys():
            self.tex_names[key] = sp.symbols(self.parent.tex_names[key])

        # Vars
        for vname, var in self.parent.vars.items():
            self.inputs_dict[vname] = sp.symbols(f'{vname}')
            self.tex_map[rf"\b{vname}\b"] = rf'{var.tex_name}'

        # RParams
        for rpname, rparam in self.parent.rparams.items():
            self.inputs_dict[rpname] = sp.symbols(f'{rparam.name}')
            self.tex_map[rf"\b{rpname}\b"] = f'{rparam.tex_name}'

        # Routine Services
        for sname, service in self.parent.services.items():
            tmp = sp.symbols(f'{service.name}')
            self.services_dict[sname] = tmp
            self.inputs_dict[sname] = tmp
            self.tex_map[rf"\b{sname}\b"] = f'{service.tex_name}'

        # Expressions
        for ename, expr in self.parent.exprs.items():
            self.inputs_dict[ename] = sp.symbols(f'{ename}')
            self.tex_map[rf"\b{ename}\b"] = f'{expr.tex_name}'

        # store tex names defined in `self.config`
        for key in self.config.as_dict():
            tmp = sp.symbols(key)
            if key not in self.config.tex_names.keys():
                logger.debug(f'No tex name for config.{key}')
                self.tex_map[rf"\b{key}\b"] = key
            else:
                self.tex_map[rf"\b{key}\b"] = self.config.tex_names[key]
            self.inputs_dict[key] = tmp
            if key in self.config.tex_names:
                self.tex_names[tmp] = sp.Symbol(self.config.tex_names[key])

        # store tex names for pretty printing replacement later
        for var in self.inputs_dict:
            if var in self.parent.__dict__ and self.parent.__dict__[var].tex_name is not None:
                self.tex_names[sp.symbols(var)] = sp.symbols(self.parent.__dict__[var].tex_name)

        # additional variables by conventions that are defined in ``BaseRoutine``
        self.inputs_dict['sys_f'] = sp.symbols('sys_f')
        self.inputs_dict['sys_mva'] = sp.symbols('sys_mva')

        self.parent._syms = True
        _, s = elapsed(t)

        logger.debug(f' - Symbols generated in {s}')
        return self.parent._syms

    def _check_expr_symbols(self, expr):
        """
        Check if expression contains unknown symbols.
        """
        fs = expr.free_symbols
        for item in fs:
            if item not in self.inputs_dict.values():
                raise ValueError(f'{self.class_name} expression "{expr}" contains unknown symbol "{item}"')

        fs = sorted(fs, key=lambda s: s.name)
        return fs

    def generate_pretty_print(self):
        """
        Generate pretty print math formulation.
        """
        raise NotImplementedError
