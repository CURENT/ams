"""
Symbolic processor class for AMS routines.

This module is revised from ``andes.core.symprocessor``.
"""

import logging  # NOQA
from collections import OrderedDict  # NOQA

import sympy as sp  # NOQA

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
    x: sympy.Matrix
        variables pretty print
    c : sympy.Matrix
        pretty print of variables coefficients
    Aub : sympy.Matrix
        Aub pretty print
    Aeq : sympy.Matrix
        Aeq pretty print
    bub : sympy.Matrix
        pretty print of inequality upper bound
    beq : sympy.Matrix
        pretty print of equality bound
    lb : sympy.Matrix
        pretty print of variables lower bound
    ub : sympy.Matrix
        pretty print of variables upper bound
    inputs_dict : OrderedDict
        All possible symbols in equations, including variables, parameters, and
        config flags.
    vars_dict : OrderedDict
        variable-only symbols, which are useful when getting the Jacobian matrices.
    """

    def __init__(self, parent):
        self.parent = parent
        self.inputs_dict = OrderedDict()
        self.vars_dict = OrderedDict()
        self.vars_list = list()       # list of variable symbols, corresponding to `self.xy`
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
            (r'\bproblem\b', f'{lang}.Problem'),
            (r'\bmultiply\b', f'{lang}.multiply'),
            (r'\bmul\b', f'{lang}.multiply'),  # alias for multiply
            (r'\bvstack\b', f'{lang}.vstack'),
            (r'\bnorm\b', f'{lang}.norm'),
            (r'\bpos\b', f'{lang}.pos'),
            (r'\bpower\b', f'{lang}.power'),
            (r'\bsign\b', f'{lang}.sign'),
        ])

        self.tex_map = OrderedDict([
            (r'\*\*(\d+)', '^{\\1}'),
            (r'\b(\w+)\s*\*\s*(\w+)\b', r'\1 \2'),
            (r'\@', r' '),
            (r'dot', r' '),
            (r'multiply\(([^,]+), ([^)]+)\)', r'\1 \2'),
            (r'\bnp.linalg.pinv(\d+)', r'\1^{\-1}'),
            (r'\bpos\b', 'F^{+}'),
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
            self.vars_dict[vname] = tmp
            self.inputs_dict[vname] = tmp
            self.sub_map[rf"\b{vname}\b"] = f"self.om.{vname}"
            self.tex_map[rf"\b{vname}\b"] = rf'{var.tex_name}'

        # RParams
        for rpname, rparam in self.parent.rparams.items():
            tmp = sp.symbols(f'{rparam.name}')
            self.inputs_dict[rpname] = tmp
            self.sub_map[rf"\b{rpname}\b"] = f'self.om.rtn.{rpname}.v'
            self.tex_map[rf"\b{rpname}\b"] = f'{rparam.tex_name}'

        # Routine Services
        for sname, service in self.parent.services.items():
            tmp = sp.symbols(f'{service.name}')
            self.services_dict[sname] = tmp
            self.inputs_dict[sname] = tmp
            self.sub_map[rf"\b{sname}\b"] = f'self.om.rtn.{sname}.v'
            self.tex_map[rf"\b{sname}\b"] = f'{service.tex_name}'

        # store tex names defined in `self.config`
        for key in self.config.as_dict():
            tmp = sp.symbols(key)
            self.sub_map[rf"\b{key}\b"] = f'self.rtn.config.{key}'
            self.tex_map[rf"\b{key}\b"] = f'{key}'
            self.inputs_dict[key] = tmp
            if key in self.config.tex_names:
                self.tex_names[tmp] = sp.Symbol(self.config.tex_names[key])

        # NOTE: hard-coded config 't' tex name as 'T_{cfg}' for clarity in doc 
        self.tex_map['\\bt\\b'] = 'T_{cfg}'
        self.tex_map['\\bcul\\b'] = 'c_{ul, cfg}'

        # store tex names for pretty printing replacement later
        for var in self.inputs_dict:
            if var in self.parent.__dict__ and self.parent.__dict__[var].tex_name is not None:
                self.tex_names[sp.symbols(var)] = sp.symbols(self.parent.__dict__[var].tex_name)

        # additional variables by conventions that are defined in ``BaseRoutine``
        self.inputs_dict['sys_f'] = sp.symbols('sys_f')
        self.inputs_dict['sys_mva'] = sp.symbols('sys_mva')

        self.vars_list = list(self.vars_dict.values())  # useful for ``.jacobian()``

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
