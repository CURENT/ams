"""
Module for optimization OModel.
"""
import logging

from typing import Any
from collections import OrderedDict

from andes.utils.misc import elapsed

import cvxpy as cp

from ams.opt.optzbase import ensure_symbols, ensure_mats_and_parsed


logger = logging.getLogger(__name__)


class OModelBase:
    """
    Template class for optimization models.
    """

    def __init__(self, routine):
        self.rtn = routine
        self.prob = None
        self.exprs = OrderedDict()
        self.params = OrderedDict()
        self.vars = OrderedDict()
        self.constrs = OrderedDict()
        self.obj = None
        self.parsed = False
        self.evaluated = False
        self.finalized = False

    @property
    def initialized(self):
        """
        Return the initialization status.
        """
        return self.parsed and self.evaluated and self.finalized

    def parse(self, force=False):
        self.parsed = True
        return self.parsed

    def _evaluate_params(self):
        return True

    def _evaluate_vars(self):
        return True

    def _evaluate_constrs(self):
        return True

    def _evaluate_obj(self):
        return True

    def _evaluate_exprs(self):
        return True

    def _evaluate_exprcs(self):
        return True

    def evaluate(self, force=False):
        self._evaluate_params()
        self._evaluate_vars()
        self._evaluate_exprs()
        self._evaluate_constrs()
        self._evaluate_obj()
        self._evaluate_exprcs()
        self.evaluated = True
        return self.evaluated

    def finalize(self, force=False):
        self.finalized = True
        return True

    def init(self, force=False):
        self.parse(force)
        self.evaluate(force)
        self.finalize(force)
        return self.initialized

    @property
    def class_name(self):
        return self.__class__.__name__

    def _register_attribute(self, key, value):
        """
        Register a pair of attributes to OModel instance.

        Called within ``__setattr__``, this is where the magic happens.
        Subclass attributes are automatically registered based on the variable type.
        """
        if isinstance(value, cp.Variable):
            self.vars[key] = value
        elif isinstance(value, cp.Constraint):
            self.constrs[key] = value
        elif isinstance(value, cp.Parameter):
            self.params[key] = value
        elif isinstance(value, cp.Expression):
            self.exprs[key] = value

    def __setattr__(self, name: str, value: Any):
        super().__setattr__(name, value)
        self._register_attribute(name, value)

    def update(self, params):
        return True

    def __repr__(self) -> str:
        return f'{self.rtn.class_name}.{self.__class__.__name__} at {hex(id(self))}'


class OModel(OModelBase):
    """
    Base class for optimization models.

    Parameters
    ----------
    routine: Routine
        Routine that to be modeled.

    Attributes
    ----------
    prob: cvxpy.Problem
        Optimization model.
    exprs: OrderedDict
        Expressions registry.
    params: OrderedDict
        Parameters registry.
    vars: OrderedDict
        Decision variables registry.
    constrs: OrderedDict
        Constraints registry.
    obj: Objective
        Objective function.
    initialized: bool
        Flag indicating if the model is initialized.
    parsed: bool
        Flag indicating if the model is parsed.
    evaluated: bool
        Flag indicating if the model is evaluated.
    finalized: bool
        Flag indicating if the model is finalized.
    """

    def __init__(self, routine):
        OModelBase.__init__(self, routine)

    @ensure_symbols
    def parse(self, force=False):
        """
        Parse the optimization model from the symbolic description.

        This method should be called after the routine symbols are generated
        `self.rtn.syms.generate_symbols()`. It parses the following components
        of the optimization model: parameters, decision variables, constraints,
        objective function, and expressions.

        Parameters
        ----------
        force : bool, optional
            Flag indicating if to force the parsing.

        Returns
        -------
        bool
            Returns True if the parsing is successful, False otherwise.
        """
        if self.parsed and not force:
            logger.debug("Model is already parsed.")
            return self.parsed

        t, _ = elapsed()
        logger.warning(f'Parsing OModel for <{self.rtn.class_name}>')
        # --- add expressions ---
        for key, val in self.rtn.exprs.items():
            val.parse()

        # --- add RParams and Services as parameters ---
        for key, val in self.rtn.params.items():
            if not val.no_parse:
                val.parse()

        # --- add decision variables ---
        for key, val in self.rtn.vars.items():
            val.parse()

        # --- add constraints ---
        for key, val in self.rtn.constrs.items():
            val.parse()

        # --- add ExpressionCalcs ---
        for key, val in self.rtn.exprcs.items():
            val.parse()

        # --- parse objective functions ---
        if self.rtn.obj is not None:
            try:
                self.rtn.obj.parse()
            except Exception as e:
                raise Exception(f"Failed to parse Objective <{self.rtn.obj.name}>.\n{e}")
        elif self.rtn.class_name not in ['DCPF0']:
            logger.warning(f"{self.rtn.class_name} has no objective function!")
            self.parsed = False
            return self.parsed

        # --- parse expressions ---
        for key, val in self.rtn.exprs.items():
            try:
                val.parse()
            except Exception as e:
                raise Exception(f"Failed to parse ExpressionCalc <{key}>.\n{e}")
        _, s = elapsed(t)
        logger.debug(f"  -> Parsed in {s}")

        self.parsed = True
        return self.parsed

    def _evaluate_params(self):
        """
        Evaluate the parameters.
        """
        for key, val in self.rtn.params.items():
            try:
                val.evaluate()
                setattr(self, key, val.optz)
            except Exception as e:
                raise Exception(f"Failed to evaluate Param <{key}>.\n{e}")

    def _evaluate_vars(self):
        """
        Evaluate the decision variables.
        """
        for key, val in self.rtn.vars.items():
            try:
                val.evaluate()
                setattr(self, key, val.optz)
            except Exception as e:
                raise Exception(f"Failed to evaluate Var <{key}>.\n{e}")

    def _evaluate_constrs(self):
        """
        Evaluate the constraints.
        """
        for key, val in self.rtn.constrs.items():
            try:
                val.evaluate()
                setattr(self, key, val.optz)
            except Exception as e:
                raise Exception(f"Failed to evaluate Constr <{key}>.\n{e}")

    def _evaluate_obj(self):
        """
        Evaluate the objective function.
        """
        # NOTE: since we already have the attribute `obj`,
        # we can update it rather than setting it
        if self.rtn.obj is not None:
            self.rtn.obj.evaluate()
            self.obj = self.rtn.obj.optz

    def _evaluate_exprs(self):
        """
        Evaluate the expressions.
        """
        for key, val in self.rtn.exprs.items():
            try:
                val.evaluate()
                setattr(self, key, val.optz)
            except Exception as e:
                raise Exception(f"Failed to evaluate Expression <{key}>.\n{e}")

    def _evaluate_exprcs(self):
        """
        Evaluate the expressions.
        """
        for key, val in self.rtn.exprcs.items():
            try:
                val.evaluate()
            except Exception as e:
                raise Exception(f"Failed to evaluate ExpressionCalc <{key}>.\n{e}")

    @ensure_mats_and_parsed
    def evaluate(self, force=False):
        """
        Evaluate the optimization model.

        This method should be called after `self.parse()`. It evaluates the following
        components of the optimization model: parameters, decision variables, constraints,
        objective function, and expressions.

        Parameters
        ----------
        force : bool, optional
            Flag indicating if to force the evaluation

        Returns
        -------
        bool
            Returns True if the evaluation is successful, False otherwise.
        """
        if self.evaluated and not force:
            logger.debug("Model is already evaluated.")
            return self.evaluated
        logger.warning(f"Evaluating OModel for <{self.rtn.class_name}>")
        t, _ = elapsed()

        # NOTE: should evaluate in sequence
        self._evaluate_params()
        self._evaluate_vars()
        self._evaluate_exprs()
        self._evaluate_constrs()
        self._evaluate_obj()
        self._evaluate_exprcs()

        self.evaluated = True
        _, s = elapsed(t)
        logger.debug(f" -> Evaluated in {s}")
        return self.evaluated

    def finalize(self, force=False):
        """
        Finalize the optimization model.

        This method should be called after `self.evaluate()`. It assemble the optimization
        problem from the evaluated components.

        Returns
        -------
        bool
            Returns True if the finalization is successful, False otherwise.
        """
        # NOTE: for power flow type, we skip the finalization
        if self.rtn.class_name in ['DCPF0']:
            self.finalized = True
            return self.finalized
        if self.finalized and not force:
            logger.debug("Model is already finalized.")
            return self.finalized
        logger.warning(f"Finalizing OModel for <{self.rtn.class_name}>")
        t, _ = elapsed()

        # Collect constraints that are not disabled
        constrs_add = [val.optz for key, val in self.rtn.constrs.items(
        ) if not val.is_disabled and val is not None]
        # Construct the problem using cvxpy.Problem
        self.prob = cp.Problem(self.obj, constrs_add)

        _, s = elapsed(t)
        logger.debug(f" -> Finalized in {s}")
        self.finalized = True
        return self.finalized

    def init(self, force=False):
        """
        Set up the optimization model from the symbolic description.

        This method initializes the optimization model by parsing decision variables,
        constraints, and the objective function from the associated routine.

        Parameters
        ----------
        force : bool, optional
            Flag indicating if to force the OModel initialization.
            If True, following methods will be called by force: `self.parse()`,
            `self.evaluate()`, `self.finalize()`

        Returns
        -------
        bool
            Returns True if the setup is successful, False otherwise.
        """
        if self.initialized and not force:
            logger.debug("OModel is already initialized.")
            return self.initialized

        t, _ = elapsed()

        self.parse(force=force)
        self.evaluate(force=force)
        self.finalize(force=force)

        _, s = elapsed(t)
        logger.debug(f"OModel for <{self.rtn.class_name}> initialized in {s}")

        return self.initialized

    @property
    def class_name(self):
        """
        Return the class name
        """
        return self.__class__.__name__

    def _register_attribute(self, key, value):
        """
        Register a pair of attributes to OModel instance.

        Called within ``__setattr__``, this is where the magic happens.
        Subclass attributes are automatically registered based on the variable type.
        """
        if isinstance(value, cp.Variable):
            self.vars[key] = value
        elif isinstance(value, cp.Constraint):
            self.constrs[key] = value
        elif isinstance(value, cp.Parameter):
            self.params[key] = value
        elif isinstance(value, cp.Expression):
            self.exprs[key] = value

    def __setattr__(self, name: str, value: Any):
        super().__setattr__(name, value)
        self._register_attribute(name, value)

    def update(self, params):
        """
        Update the Parameter values.

        Parameters
        ----------
        params: list
            List of parameters to be updated.
        """
        for param in params:
            param.update()
        return True

    def __repr__(self) -> str:
        return f'{self.rtn.class_name}.{self.__class__.__name__} at {hex(id(self))}'
