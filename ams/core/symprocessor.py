"""
Symbolic processor class for AMS routines.

This module is revised from ``andes.core.symprocessor``.
"""

import logging
from collections import OrderedDict

import sympy as sp

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
        self.config = parent.config
        self.class_name = parent.class_name
        self.tex_names = OrderedDict()

        lang = "cp"  # TODO: might need to be generalized to other solvers
        self.sub_map = OrderedDict([
            (r'\b(\w+)\s*\*\s*(\w+)\b', r'\1 @ \2'),
            (r'\bsum\b', f'{lang}.sum'),  # only used for CVXPY
            (r'\bvar\b', f'{lang}.Variable'),  # only used for CVXPY
            # (r'\bmin\b', f'{solver}.Minimize'),
            # ('pg', 'self.pg'),
            # ('c2', 'self.routine.c2.v'),
            # ('c1', 'self.routine.c1.v'),
            # ('c0', 'self.routine.c0.v'),
            ])

    def generate_symbols(self):
        """
        Generate symbols for all variables.
        """
        logger.debug(f'- Generating symbols for {self.parent.class_name}')
        # process tex_names defined in routines
        # -----------------------------------------------------------
        for key in self.parent.tex_names.keys():
            self.tex_names[key] = sp.symbols(self.parent.tex_names[key])

        # RAlgebs
        for raname, ralgeb in self.parent.ralgebs.items():
            tmp = sp.symbols(f'{ralgeb.name}')
            # tmp = sp.symbols(ralgeb.name)
            self.vars_dict[raname] = tmp
            self.inputs_dict[raname] = tmp
            self.sub_map[raname] = f'self.{raname}'

        # RParams
        for rpname, rparam in self.parent.rparams.items():
            tmp = sp.symbols(f'{rparam.name}')
            self.inputs_dict[rpname] = tmp
            self.sub_map[rpname] = f'self.routine.{rpname}.v'

        # store tex names defined in `self.config`
        for key in self.config.as_dict():
            tmp = sp.symbols(key)
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

        self.vars_list = list(self.vars_dict.values())  # useful for ``.jacobian()``

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
        logger.debug("- Generating pretty prints for %s", self.class_name)

        # equation symbols for pretty printing
        self.c = sp.Matrix([])
        self.bub, self.beq = sp.Matrix([]), sp.Matrix([])
        self.lb, self.ub = sp.Matrix([]), sp.Matrix([])

        try:
            self.x = sp.Matrix(list(self.vars_dict.values())).subs(self.tex_names)
        except TypeError as e:
            logger.error("Error while substituting tex_name for variables.")
            logger.error("Variable names might have conflicts with SymPy functions.")
            raise e

        # get pretty printing equations by substituting symbols
        self.Aub = self.Aub_matrix.subs(self.tex_names)
        self.Aeq = self.Aeq_matrix.subs(self.tex_names)

        # --- disabled part --- not understand yet, seems not necessary in AMS?
        # store latex strings
        # nub = len(self.Aub)
        # neq = len(self.Aeq)
        # self.calls.x_latex = [sp.latex(item) for item in self.xy[:nx]]
        # self.calls.y_latex = [sp.latex(item) for item in self.xy[nx:nx + ny]]

        # self.calls.f_latex = [sp.latex(item) for item in self.f]
        # self.calls.g_latex = [sp.latex(item) for item in self.g]
        # self.calls.s_latex = [sp.latex(item) for item in self.s]

        # self.df = self.df_sparse.subs(self.tex_names)
        # self.dg = self.dg_sparse.subs(self.tex_names)
        # --- end ---
