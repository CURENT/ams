"""
Module for optimization modeling.
"""
import logging

from typing import Any, Optional, Union
from collections import OrderedDict
import re

import numpy as np

from andes.core.common import Config
from andes.utils.misc import elapsed

import cvxpy as cp

from ams.utils import pretty_long_message
from ams.shared import sps

logger = logging.getLogger(__name__)

_prefix = r" - --------------> | "
_max_length = 80


def ensure_symbols(func):
    """
    Decorator to ensure that symbols are generated before parsing.
    If not, it runs self.rtn.syms.generate_symbols().

    Designed to be used on the `parse` method of the optimization elements (`OptzBase`)
    and optimization model (`OModel`), i.e., `Var`, `Param`, `Constraint`, `Objective`,
    and `ExpressionCalc`.

    Note:
    -----
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
    Decorator to ensure that system matrices are built and OModel is parsed
    before evaluation. If not, it runs the necessary methods to initialize them.

    Designed to be used on the `evaluate` method of the optimization elements (`OptzBase`)
    and optimization model (`OModel`), i.e., `Var`, `Param`, `Constraint`, `Objective`,
    and `ExpressionCalc`.

    Note:
    -----
    Evaluation before matrices building and parsing can run into errors. Ensure that
    system matrices are built and OModel is parsed before calling the `evaluate` method.
    """

    def wrapper(self, *args, **kwargs):
        try:
            if not self.rtn.system.mats.initialized:
                logger.debug("System matrices are not built yet. Building now...")
                self.rtn.system.mats.build()
            if isinstance(self, (OptzBase, Var, Param, Constraint, Objective)):
                if not self.om.parsed:
                    logger.debug("OModel is not parsed yet. Parsing now...")
                    self.om.parse()
            elif isinstance(self, OModel):
                if not self.parsed:
                    logger.debug("OModel is not parsed yet. Parsing now...")
                    self.parse()
        except Exception as e:
            logger.error(f"Error during initialization or parsing: {e}")
            raise
        return func(self, *args, **kwargs)
    return wrapper


class OptzBase:
    """
    Base class for optimization elements, e.g., Var and Constraint.

    Parameters
    ----------
    name : str, optional
        Name.
    info : str, optional
        Descriptive information

    Attributes
    ----------
    rtn : ams.routines.Routine
        The owner routine instance.

    Note:
    -----
    Ensure that symbols are generated before calling the `parse` method. Parsing
    before symbol generation can give wrong results.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 ):
        self.om = None
        self.name = name
        self.info = info
        self.unit = unit
        self.is_disabled = False
        self.rtn = None
        self.optz = None  # corresponding optimization element
        self.code = None

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


class ExpressionCalc(OptzBase):
    """
    Expression for calculation.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 var: Optional[str] = None,
                 e_str: Optional[str] = None,
                 ):
        OptzBase.__init__(self, name=name, info=info, unit=unit)
        self.optz = None
        self.var = var
        self.e_str = e_str
        self.code = None

    @ensure_symbols
    def parse(self):
        """
        Parse the Expression.
        """
        # parse the expression str
        sub_map = self.om.rtn.syms.sub_map
        code_expr = self.e_str
        for pattern, replacement in sub_map.items():
            try:
                code_expr = re.sub(pattern, replacement, code_expr)
            except Exception as e:
                raise Exception(f"Error in parsing expr <{self.name}>.\n{e}")
        # store the parsed expression str code
        self.code = code_expr
        msg = f" - ExpressionCalc <{self.name}>: {self.e_str}"
        logger.debug(pretty_long_message(msg, _prefix, max_length=_max_length))
        return True

    @ensure_mats_and_parsed
    def evaluate(self):
        """
        Evaluate the expression.
        """
        msg = f" - Expression <{self.name}>: {self.code}"
        logger.debug(pretty_long_message(msg, _prefix, max_length=_max_length))
        try:
            local_vars = {'self': self, 'np': np, 'cp': cp}
            self.optz = self._evaluate_expression(self.code, local_vars=local_vars)
        except Exception as e:
            raise Exception(f"Error in evaluating expr <{self.name}>.\n{e}")
        return True

    def _evaluate_expression(self, code, local_vars=None):
        """
        Helper method to evaluate the expression code.

        Parameters
        ----------
        code : str
            The code string representing the expression.

        Returns
        -------
        cp.Expression
            The evaluated cvxpy expression.
        """
        return eval(code, {}, local_vars)

    @property
    def v(self):
        """
        Return the CVXPY expression value.
        """
        if self.optz is None:
            return None
        else:
            return self.optz.value

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.name}'


class Param(OptzBase):
    """
    Base class for parameters used in a routine.

    Parameters
    ----------
    no_parse: bool, optional
        True to skip parsing the parameter.
    nonneg: bool, optional
        True to set the parameter as non-negative.
    nonpos: bool, optional
        True to set the parameter as non-positive.
    cplx: bool, optional
        True to set the parameter as complex, avoiding the use of `complex`.
    imag: bool, optional
        True to set the parameter as imaginary.
    symmetric: bool, optional
        True to set the parameter as symmetric.
    diag: bool, optional
        True to set the parameter as diagonal.
    hermitian: bool, optional
        True to set the parameter as hermitian.
    boolean: bool, optional
        True to set the parameter as boolean.
    integer: bool, optional
        True to set the parameter as integer.
    pos: bool, optional
        True to set the parameter as positive.
    neg: bool, optional
        True to set the parameter as negative.
    sparse: bool, optional
        True to set the parameter as sparse.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 no_parse: Optional[bool] = False,
                 nonneg: Optional[bool] = False,
                 nonpos: Optional[bool] = False,
                 cplx: Optional[bool] = False,
                 imag: Optional[bool] = False,
                 symmetric: Optional[bool] = False,
                 diag: Optional[bool] = False,
                 hermitian: Optional[bool] = False,
                 boolean: Optional[bool] = False,
                 integer: Optional[bool] = False,
                 pos: Optional[bool] = False,
                 neg: Optional[bool] = False,
                 sparse: Optional[bool] = False,
                 ):
        OptzBase.__init__(self, name=name, info=info, unit=unit)
        self.no_parse = no_parse  # True to skip parsing the parameter
        self.sparse = sparse

        self.config = Config(name=self.class_name)  # `config` that can be exported

        self.config.add(OrderedDict((('nonneg', nonneg),
                                     ('nonpos', nonpos),
                                     ('complex', cplx),
                                     ('imag', imag),
                                     ('symmetric', symmetric),
                                     ('diag', diag),
                                     ('hermitian', hermitian),
                                     ('boolean', boolean),
                                     ('integer', integer),
                                     ('pos', pos),
                                     ('neg', neg),
                                     )))

    @ensure_symbols
    def parse(self):
        """
        Parse the parameter.

        Returns
        -------
        bool
            Returns True if the parsing is successful, False otherwise.
        """
        sub_map = self.om.rtn.syms.sub_map
        code_param = "param(**config)"
        for pattern, replacement, in sub_map.items():
            try:
                code_param = re.sub(pattern, replacement, code_param)
            except Exception as e:
                raise Exception(f"Error in parsing param <{self.name}>.\n{e}")
        self.code = code_param
        return True

    @ensure_mats_and_parsed
    def evaluate(self):
        """
        Evaluate the parameter.
        """
        if self.no_parse:
            return True

        config = self.config.as_dict()
        try:
            msg = f"Parameter <{self.name}> is set as sparse, "
            msg += "but the value is not sparse."
            if self.sparse:
                self.v = sps.csr_matrix(self.v)

            # Create the cvxpy.Parameter object
            self.optz = cp.Parameter(shape=self.v.shape, **config)
            self.optz.value = self.v
        except ValueError:
            msg = f"Parameter <{self.name}> has non-numeric value, "
            msg += "set `no_parse=True`."
            logger.warning(msg)
            self.no_parse = True
            return True
        except Exception as e:
            raise Exception(f"Error in evaluating param <{self.name}>.\n{e}")
        return True

    def update(self):
        """
        Update the Parameter value.
        """
        # NOTE: skip no_parse parameters
        if self.optz is None:
            return None
        self.optz.value = self.v
        return True

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.name}'


class Var(OptzBase):
    """
    Base class for variables used in a routine.

    When `horizon` is provided, the variable will be expanded to a matrix,
    where rows are indexed by the source variable index and columns are
    indexed by the horizon index.

    Parameters
    ----------
    info : str, optional
        Descriptive information
    unit : str, optional
        Unit
    tex_name : str
        LaTeX-formatted variable symbol. Defaults to the value of ``name``.
    name : str, optional
        Variable name. One should typically assigning the name directly because
        it will be automatically assigned by the model. The value of ``name``
        will be the symbol name to be used in expressions.
    src : str, optional
        Source variable name. Defaults to the value of ``name``.
    model : str, optional
        Name of the owner model or group.
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
    boolean : bool, optional
        Boolean variable
    integer : bool, optional
        Integer variable
    pos : bool, optional
        Positive variable
    neg : bool, optional
        Negative variable

    Attributes
    ----------
    a : np.ndarray
        Variable address.
    _v : np.ndarray
        Local-storage of the variable value.
    rtn : ams.routines.Routine
        The owner routine instance.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 src: Optional[str] = None,
                 unit: Optional[str] = None,
                 model: Optional[str] = None,
                 shape: Optional[Union[tuple, int]] = None,
                 v0: Optional[str] = None,
                 horizon=None,
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
                 neg: Optional[bool] = False,
                 ):
        self.name = name
        self.info = info
        self.unit = unit

        self.tex_name = tex_name if tex_name else name
        # instance of the owner Model
        self.owner = None
        # variable internal index inside a model (assigned in run time)
        self.id = None
        OptzBase.__init__(self, name=name, info=info, unit=unit)
        self.src = src
        self.is_group = False
        self.model = model  # indicate if this variable is a group variable
        self.owner = None  # instance of the owner model or group
        self.v0 = v0
        self.horizon = horizon
        self._shape = shape
        self._v = None
        self.a: np.ndarray = np.array([], dtype=int)

        self.config = Config(name=self.class_name)  # `config` that can be exported

        self.config.add(OrderedDict((('nonneg', nonneg),
                                     ('nonpos', nonpos),
                                     ('complex', cplx),
                                     ('imag', imag),
                                     ('symmetric', symmetric),
                                     ('diag', diag),
                                     ('psd', psd),
                                     ('nsd', nsd),
                                     ('hermitian', hermitian),
                                     ('boolean', boolean),
                                     ('integer', integer),
                                     ('pos', pos),
                                     ('neg', neg),
                                     )))

    @property
    def v(self):
        """
        Return the CVXPY variable value.
        """
        if self.optz is None:
            return None
        if self.optz.value is None:
            try:
                shape = self.optz.shape
                return np.zeros(shape)
            except AttributeError:
                return None
        else:
            return self.optz.value

    @v.setter
    def v(self, value):
        if self.optz is None:
            logger.info(f"Variable <{self.name}> is not initialized yet.")
        else:
            self.optz.value = value

    def get_idx(self):
        if self.is_group:
            return self.owner.get_idx()
        elif self.owner is None:
            logger.info(f'Variable <{self.name}> has no owner.')
            return None
        else:
            return self.owner.idx.v

    @ensure_symbols
    def parse(self):
        """
        Parse the variable.
        """
        sub_map = self.om.rtn.syms.sub_map
        # NOTE: number of rows is the size of the source variable
        if self.owner is not None:
            nr = self.owner.n
            nc = 0
            if self.horizon:
                # NOTE: numer of columns is the horizon if exists
                nc = int(self.horizon.n)
                shape = (nr, nc)
            else:
                shape = (nr,)
        elif isinstance(self._shape, int):
            shape = (self._shape,)
            nr = shape
            nc = 0
        elif isinstance(self._shape, tuple):
            shape = self._shape
            nr = shape[0]
            nc = shape[1] if len(shape) > 1 else 0
        else:
            raise ValueError(f"Invalid shape {self._shape}.")
        code_var = f"var({shape}, **config)"
        logger.debug(f" - Var <{self.name}>: {self.code}")
        for pattern, replacement, in sub_map.items():
            try:
                code_var = re.sub(pattern, replacement, code_var)
            except Exception as e:
                raise Exception(f"Error in parsing var <{self.name}>.\n{e}")
        self.code = code_var
        return True

    @ensure_mats_and_parsed
    def evaluate(self):
        """
        Evaluate the variable.

        Returns
        -------
        bool
            Returns True if the evaluation is successful, False otherwise.
        """
        # NOTE: in CVXPY, Config only allow lower case letters
        config = {}  # used in `self.code`
        for k, v in self.config.as_dict().items():
            if k == 'psd':
                config['PSD'] = v
            elif k == 'nsd':
                config['NSD'] = v
            elif k == 'bool':
                config['boolean'] = v
            else:
                config[k] = v
        msg = f" - Var <{self.name}>: {self.code}"
        logger.debug(pretty_long_message(msg, _prefix, max_length=_max_length))
        try:
            local_vars = {'self': self, 'config': config, 'cp': cp}
            self.optz = eval(self.code, {}, local_vars)
        except Exception as e:
            raise Exception(f"Error in evaluating var <{self.name}>.\n{e}")
        return True

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.owner.__class__.__name__}.{self.name}'


class Constraint(OptzBase):
    """
    Base class for constraints.

    This class is used as a template for defining constraints. Each
    instance of this class represents a single constraint.

    Parameters
    ----------
    name : str, optional
        A user-defined name for the constraint.
    e_str : str, optional
        A mathematical expression representing the constraint.
    info : str, optional
        Additional informational text about the constraint.
    is_eq : str, optional
        Flag indicating if the constraint is an equality constraint. False indicates
        an inequality constraint in the form of `<= 0`.

    Attributes
    ----------
    is_disabled : bool
        Flag indicating if the constraint is disabled, False by default.
    rtn : ams.routines.Routine
        The owner routine instance.
    is_disabled : bool, optional
        Flag indicating if the constraint is disabled, False by default.
    dual : float, optional
        The dual value of the constraint.
    code : str, optional
        The code string for the constraint
    """

    def __init__(self,
                 name: Optional[str] = None,
                 e_str: Optional[str] = None,
                 info: Optional[str] = None,
                 is_eq: Optional[bool] = False,
                 ):
        OptzBase.__init__(self, name=name, info=info)
        self.e_str = e_str
        self.is_eq = is_eq
        self.is_disabled = False
        self.dual = None
        self.code = None

    @ensure_symbols
    def parse(self):
        """
        Parse the constraint.
        """
        # parse the expression str
        sub_map = self.om.rtn.syms.sub_map
        code_constr = self.e_str
        for pattern, replacement in sub_map.items():
            try:
                code_constr = re.sub(pattern, replacement, code_constr)
            except TypeError as e:
                raise TypeError(f"Error in parsing constr <{self.name}>.\n{e}")
        # parse the constraint type
        code_constr += " == 0" if self.is_eq else " <= 0"
        # store the parsed expression str code
        self.code = code_constr
        msg = f" - Constr <{self.name}>: {self.e_str}"
        logger.debug(pretty_long_message(msg, _prefix, max_length=_max_length))
        return True

    def _evaluate_expression(self, code, local_vars=None):
        """
        Helper method to evaluate the expression code.

        Parameters
        ----------
        code : str
            The code string representing the expression.

        Returns
        -------
        cp.Expression
            The evaluated cvxpy expression.
        """
        return eval(code, {}, local_vars)

    @ensure_mats_and_parsed
    def evaluate(self):
        """
        Evaluate the constraint.
        """
        msg = f" - Constr <{self.name}>: {self.code}"
        logger.debug(pretty_long_message(msg, _prefix, max_length=_max_length))
        try:
            local_vars = {'self': self, 'cp': cp, 'sub_map': self.om.rtn.syms.val_map}
            self.optz = self._evaluate_expression(self.code, local_vars=local_vars)
        except Exception as e:
            raise Exception(f"Error in evaluating constr <{self.name}>.\n{e}")

    def __repr__(self):
        enabled = 'OFF' if self.is_disabled else 'ON'
        out = f"{self.class_name}: {self.name} [{enabled}]"
        return out

    @property
    def e(self):
        """
        Return the calculated constraint LHS value.
        Note that `v` should be used primarily as it is obtained
        from the solver directly.

        `e` is for debugging purpose. For a successfully solved problem,
        `e` should equal to `v`. However, when a problem is infeasible
        or unbounded, `e` can be used to check the constraint LHS value.
        """
        if self.code is None:
            logger.info(f"Constraint <{self.name}> is not parsed yet.")
            return None

        val_map = self.om.rtn.syms.val_map
        code = self.code
        for pattern, replacement in val_map.items():
            try:
                code = re.sub(pattern, replacement, code)
            except TypeError as e:
                raise TypeError(e)

        try:
            logger.debug(pretty_long_message(f"Value code: {code}",
                                             _prefix, max_length=_max_length))
            local_vars = {'self': self, 'np': np, 'cp': cp, 'val_map': val_map}
            return self._evaluate_expression(code, local_vars)
        except Exception as e:
            logger.error(f"Error in calculating constr <{self.name}>.\n{e}")
            return None

    @property
    def v(self):
        """
        Return the CVXPY constraint LHS value.
        """
        if self.optz is None:
            return None
        if self.optz._expr.value is None:
            try:
                shape = self._expr.shape
                return np.zeros(shape)
            except AttributeError:
                return None
        else:
            return self.optz._expr.value

    @v.setter
    def v(self, value):
        raise AttributeError("Cannot set the value of the constraint.")


class Objective(OptzBase):
    """
    Base class for objective functions.

    This class serves as a template for defining objective functions. Each
    instance of this class represents a single objective function that can
    be minimized or maximized depending on the sense ('min' or 'max').

    Parameters
    ----------
    name : str, optional
        A user-defined name for the objective function.
    e_str : str, optional
        A mathematical expression representing the objective function.
    info : str, optional
        Additional informational text about the objective function.
    sense : str, optional
        The sense of the objective function, default to 'min'.
        `min` for minimization and `max` for maximization.

    Attributes
    ----------
    v : NoneType
        The value of the objective function. It needs to be set through
        computation.
    rtn : ams.routines.Routine
        The owner routine instance.
    code : str
        The code string for the objective function.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 e_str: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 sense: Optional[str] = 'min'):
        OptzBase.__init__(self, name=name, info=info, unit=unit)
        self.e_str = e_str
        self.sense = sense
        self.code = None

    @property
    def e(self):
        """
        Return the calculated objective value.

        Note that `v` should be used primarily as it is obtained
        from the solver directly.

        `e` is for debugging purpose. For a successfully solved problem,
        `e` should equal to `v`. However, when a problem is infeasible
        or unbounded, `e` can be used to check the objective value.
        """
        if self.code is None:
            logger.info(f"Objective <{self.name}> is not parsed yet.")
            return None

        val_map = self.om.rtn.syms.val_map
        code = self.code
        for pattern, replacement in val_map.items():
            try:
                code = re.sub(pattern, replacement, code)
            except TypeError as e:
                logger.error(f"Error in parsing value for obj <{self.name}>.")
                raise e

        try:
            logger.debug(pretty_long_message(f"Value code: {code}",
                                             _prefix, max_length=_max_length))
            local_vars = {'self': self, 'np': np, 'cp': cp, 'val_map': val_map}
            return self._evaluate_expression(code, local_vars)
        except Exception as e:
            logger.error(f"Error in calculating obj <{self.name}>.\n{e}")
            return None

    @property
    def v(self):
        """
        Return the CVXPY objective value.
        """
        if self.optz is None:
            return None
        else:
            return self.optz.value

    @v.setter
    def v(self, value):
        raise AttributeError("Cannot set the value of the objective function.")

    @ensure_symbols
    def parse(self):
        """
        Parse the objective function.

        Returns
        -------
        bool
            Returns True if the parsing is successful, False otherwise.
        """
        # parse the expression str
        sub_map = self.om.rtn.syms.sub_map
        code_obj = self.e_str
        for pattern, replacement, in sub_map.items():
            try:
                code_obj = re.sub(pattern, replacement, code_obj)
            except Exception as e:
                raise Exception(f"Error in parsing obj <{self.name}>.\n{e}")
        # store the parsed expression str code
        self.code = code_obj
        if self.sense not in ['min', 'max']:
            raise ValueError(f'Objective sense {self.sense} is not supported.')
        sense = 'cp.Minimize' if self.sense == 'min' else 'cp.Maximize'
        self.code = f"{sense}({code_obj})"
        msg = f" - Objective <{self.name}>: {self.code}"
        logger.debug(pretty_long_message(msg, _prefix, max_length=_max_length))
        return True

    @ensure_mats_and_parsed
    def evaluate(self):
        """
        Evaluate the objective function.

        Returns
        -------
        bool
            Returns True if the evaluation is successful, False otherwise.
        """
        logger.debug(f" - Objective <{self.name}>: {self.e_str}")
        try:
            local_vars = {'self': self, 'cp': cp}
            self.optz = self._evaluate_expression(self.code, local_vars=local_vars)
        except Exception as e:
            raise Exception(f"Error in evaluating obj <{self.name}>.\n{e}")
        return True

    def _evaluate_expression(self, code, local_vars=None):
        """
        Helper method to evaluate the expression code.

        Parameters
        ----------
        code : str
            The code string representing the expression.

        Returns
        -------
        cp.Expression
            The evaluated cvxpy expression.
        """
        return eval(code, {}, local_vars)

    def __repr__(self):
        return f"{self.class_name}: {self.name} [{self.sense.upper()}]"


class OModel:
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
    params: OrderedDict
        Parameters.
    vars: OrderedDict
        Decision variables.
    constrs: OrderedDict
        Constraints.
    obj: Objective
        Objective function.
    initialized: bool
        Flag indicating if the model is initialized.
    parsed: bool
        Flag indicating if the model is parsed.
    evaluated: bool
        Flag indicating if the model is evaluated.
    """

    def __init__(self, routine):
        self.rtn = routine
        self.prob = None
        self.params = OrderedDict()
        self.vars = OrderedDict()
        self.constrs = OrderedDict()
        self.obj = None
        self.initialized = False
        self.parsed = False
        self.evaluated = False
        self.finalized = False

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
        # --- add RParams and Services as parameters ---
        logger.warning(f'Parsing OModel for <{self.rtn.class_name}>')
        for key, val in self.rtn.params.items():
            if not val.no_parse:
                val.parse()

        # --- add decision variables ---
        for key, val in self.rtn.vars.items():
            val.parse()

        # --- add constraints ---
        for key, val in self.rtn.constrs.items():
            val.parse()

        # --- parse objective functions ---
        if self.rtn.type != 'PF':
            if self.rtn.obj is not None:
                try:
                    self.rtn.obj.parse()
                except Exception as e:
                    raise Exception(f"Failed to parse Objective <{self.rtn.obj.name}>.\n{e}")
            else:
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
        if self.rtn.type != 'PF':
            self.rtn.obj.evaluate()
            self.obj = self.rtn.obj.optz

    def _evaluate_exprs(self):
        """
        Evaluate the expressions.
        """
        for key, val in self.rtn.exprs.items():
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

        self._evaluate_params()
        self._evaluate_vars()
        self._evaluate_constrs()
        self._evaluate_obj()
        self._evaluate_exprs()

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
        if self.rtn.type == 'PF':
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
        self.initialized = True
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
