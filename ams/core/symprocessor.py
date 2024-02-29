"""
Symbolic processor class for AMS routines.

This module is revised from ``andes.core.symprocessor``.
"""

import logging
from collections import OrderedDict

import sympy as sp

from ams.core.matprocessor import MatProcessor

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
    sub_map : dict
        Substitution map for symbolic processing.
    tex_map : dict
        Tex substitution map for documentation.
    val_map : dict
        Value substitution map for post-solving value evaluation.
    """

    def __init__(self, parent):
        self.parent = parent
        self.inputs_dict = OrderedDict()
        self.services_dict = OrderedDict()
        self.config = parent.config
        self.class_name = parent.class_name
        self.tex_names = OrderedDict()
        self.tex_map = OrderedDict()

        lang = "cp"  # TODO: might need to be generalized to other solvers
        # only used for CVXPY
        self.sub_map = OrderedDict([
            (r'\b(\w+)\s*\*\s*(\w+)\b', r'\1 @ \2'),
            (r'\b(\w+)\s+dot\s+(\w+)\b', r'\1 * \2'),
            (r' dot ', r' * '),
            (r'\bsum\b', f'{lang}.sum'),
            (r'\bvar\b', f'{lang}.Variable'),
            (r'\bparam\b', f'{lang}.Parameter'),
            (r'\bconst\b', f'{lang}.Constant'),
            (r'\bproblem\b', f'{lang}.Problem'),
            (r'\bmultiply\b', f'{lang}.multiply'),
            (r'\bmul\b', f'{lang}.multiply'),  # alias for multiply
            (r'\bvstack\b', f'{lang}.vstack'),
            (r'\bnorm\b', f'{lang}.norm'),
            (r'\bpos\b', f'{lang}.pos'),
            (r'\bpower\b', f'{lang}.power'),
            (r'\bsign\b', f'{lang}.sign'),
            (r'\bsquare\b', f'{lang}.square'),
            (r'\bquad_over_lin\b', f'{lang}.quad_over_lin'),
            (r'\bdiag\b', f'{lang}.diag'),
            (r'\bquad_form\b', f'{lang}.quad_form'),
            (r'\bsum_squares\b', f'{lang}.sum_squares'),
        ])

        self.tex_map = OrderedDict([
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
        ])

        # mapping dict for evaluating expressions
        self.val_map = OrderedDict([
            (r'\bcp.\b', 'np.'),
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
        }

    def generate_symbols(self, force_generate=False):
        """
        Generate symbols for all variables.
        """
        if not force_generate and self.parent._syms:
            return True
        logger.debug(f'- Generating symbols for {self.parent.class_name}')

        # process tex_names defined in routines
        # -----------------------------------------------------------
        for key in self.parent.tex_names.keys():
            self.tex_names[key] = sp.symbols(self.parent.tex_names[key])

        # Vars
        for vname, var in self.parent.vars.items():
            tmp = sp.symbols(f'{var.name}')
            # tmp = sp.symbols(var.name)
            self.inputs_dict[vname] = tmp
            self.sub_map[rf"\b{vname}\b"] = f"self.om.{vname}"
            self.tex_map[rf"\b{vname}\b"] = rf'{var.tex_name}'
            self.val_map[rf"\b{vname}\b"] = f"rtn.{vname}.v"

        # RParams
        for rpname, rparam in self.parent.rparams.items():
            tmp = sp.symbols(f'{rparam.name}')
            self.inputs_dict[rpname] = tmp
            sub_name = ''
            if isinstance(rparam.owner, MatProcessor):
                # sparse matrices are accessed from MatProcessor
                # otherwise, dense matrices are accessed from Routine
                if rparam.sparse:
                    sub_name = f'self.rtn.system.mats.{rpname}._v'
                else:
                    sub_name = f'self.rtn.{rpname}.v'
            elif rparam.no_parse:
                sub_name = f'self.rtn.{rpname}.v'
            else:
                sub_name = f'self.om.{rpname}'
            self.sub_map[rf"\b{rpname}\b"] = sub_name
            self.tex_map[rf"\b{rpname}\b"] = f'{rparam.tex_name}'
            if not rparam.no_parse:
                self.val_map[rf"\b{rpname}\b"] = f"rtn.{rpname}.v"

        # Routine Services
        for sname, service in self.parent.services.items():
            tmp = sp.symbols(f'{service.name}')
            self.services_dict[sname] = tmp
            self.inputs_dict[sname] = tmp
            sub_name = f'self.rtn.{sname}.v' if service.no_parse else f'self.om.{sname}'
            self.sub_map[rf"\b{sname}\b"] = sub_name
            self.tex_map[rf"\b{sname}\b"] = f'{service.tex_name}'
            if not service.no_parse:
                self.val_map[rf"\b{sname}\b"] = f"rtn.{sname}.v"

        # store tex names defined in `self.config`
        for key in self.config.as_dict():
            tmp = sp.symbols(key)
            self.sub_map[rf"\b{key}\b"] = f'self.rtn.config.{key}'
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

        return True

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
