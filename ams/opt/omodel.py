"""
Module for optimization modeling.
"""
import logging

from typing import Any, Optional, Union
from collections import OrderedDict
import re

import numpy as np
import scipy.sparse as spr

from andes.core.common import Config
from andes.utils.misc import elapsed

import cvxpy as cp

from ams.shared import sps      # NOQA

logger = logging.getLogger(__name__)


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

    def parse(self):
        """
        Parse the object.
        """
        raise NotImplementedError

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
            logger.warning('Shape info is not ready before initialziation.')
            return None

    @property
    def size(self):
        """
        Return the size.
        """
        if self.rtn.initialized:
            return self.om.__dict__[self.name].size
        else:
            logger.warning(f'<{self.rtn.class_name}> is not initialized yet.')
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

    def parse(self, no_code=True):
        """
        Parse the Expression.

        Parameters
        ----------
        no_code : bool, optional
            Flag indicating if the code should be shown, True by default.
        """
        # parse the expression str
        sub_map = self.om.rtn.syms.sub_map
        code_expr = self.e_str
        for pattern, replacement in sub_map.items():
            try:
                code_expr = re.sub(pattern, replacement, code_expr)
            except TypeError as e:
                logger.error(f"Error in parsing expr <{self.name}>.")
                raise e
        # store the parsed expression str code
        self.code = code_expr
        code_expr = "self.optz = " + code_expr
        if not no_code:
            logger.info(f"<{self.name}> code: {code_expr}")
        return True

    def evaluate(self):
        """
        Evaluate the expression.
        """
        logger.debug(f"    - Expression <{self.name}>: {self.code}")
        exec(self.code, globals(), locals())
        return True

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
                 sparse: Optional[list] = False,
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

    def parse(self):
        """
        Parse the parameter.

        Returns
        -------
        bool
            Returns True if the parsing is successful, False otherwise.
        """
        sub_map = self.om.rtn.syms.sub_map
        shape = np.shape(self.v)
        # NOTE: it seems that there is no need to use re.sub here
        code_param = f"self.optz=param(shape={shape}, **config)"
        for pattern, replacement, in sub_map.items():
            code_param = re.sub(pattern, replacement, code_param)
        self.code = code_param
        return True

    def evaluate(self):
        if self.no_parse:
            return True

        config = self.config.as_dict()  # NOQA, used in `self.code`
        exec(self.code, globals(), locals())
        try:
            msg = f"Parameter <{self.name}> is set as sparse, "
            msg += "but the value is not sparse."
            val = "self.v"
            if self.sparse:
                if not spr.issparse(self.v):
                    val = "sps.csr_matrix(self.v)"
            exec(f"self.optz.value = {val}", globals(), locals())
        except ValueError:
            msg = f"Parameter <{self.name}> has non-numeric value, "
            msg += "set `no_parse=True`."
            logger.warning(msg)
            self.no_parse = True
            return False
        except Exception as e:
            logger.error(f"Error in evaluating param <{self.name}>.")
            logger.error(f"Original error: {e}")
            return False
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
        code_var = f"self.optz=var({shape}, **config)"
        for pattern, replacement, in sub_map.items():
            code_var = re.sub(pattern, replacement, code_var)
        self.code = code_var
        return True

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
        logger.debug(f"    - Var <{self.name}>: {self.code}")
        try:
            exec(self.code, globals(), locals())
        except Exception as e:
            logger.error(f"Error in evaluating var <{self.name}>.")
            logger.error(f"Original error: {e}")
            return False
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
                 is_eq: Optional[str] = False,
                 ):
        OptzBase.__init__(self, name=name, info=info)
        self.e_str = e_str
        self.is_eq = is_eq
        self.is_disabled = False
        self.dual = None
        self.code = None

    def parse(self, no_code=True):
        """
        Parse the constraint.

        Parameters
        ----------
        no_code : bool, optional
            Flag indicating if the code should be shown, True by default.
        """
        # parse the expression str
        sub_map = self.om.rtn.syms.sub_map
        code_constr = self.e_str
        for pattern, replacement in sub_map.items():
            try:
                code_constr = re.sub(pattern, replacement, code_constr)
            except TypeError as e:
                logger.error(f"Error in parsing constr <{self.name}>.")
                raise e
        # parse the constraint type
        code_constr = "self.optz=" + code_constr
        code_constr += " == 0" if self.is_eq else " <= 0"
        # store the parsed expression str code
        self.code = code_constr
        return True

    def evaluate(self):
        """
        Evaluate the constraint.
        """
        logger.debug(f"    - Constraint <{self.name}>: {self.code}")
        try:
            exec(self.code, globals(), locals())
        except Exception as e:
            logger.error(f"Error in evaluating constr <{self.name}>.")
            logger.error(f"Original error: {e}")
            return False
        return True

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
                logger.error(f"Error in parsing value for constr <{self.name}>.")
                raise e

        try:
            logger.debug(f"Value code: {code}")
            return eval(code)
        except Exception as e:
            logger.error(f"Error in calculating constr <{self.name}>.")
            logger.error(f"Original error: {e}")
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
            logger.debug(f"Value code: {code}")
            return eval(code)
        except Exception as e:
            logger.error(f"Error in calculating obj <{self.name}>.")
            logger.error(f"Original error: {e}")
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

    def parse(self, no_code=True):
        """
        Parse the objective function.

        Parameters
        ----------
        no_code : bool, optional
            Flag indicating if the code should be shown, True by default.

        Returns
        -------
        bool
            Returns True if the parsing is successful, False otherwise.
        """
        # parse the expression str
        sub_map = self.om.rtn.syms.sub_map
        code_obj = self.e_str
        for pattern, replacement, in sub_map.items():
            code_obj = re.sub(pattern, replacement, code_obj)
        # store the parsed expression str code
        self.code = code_obj
        if self.sense not in ['min', 'max']:
            raise ValueError(f'Objective sense {self.sense} is not supported.')
        sense = 'cp.Minimize' if self.sense == 'min' else 'cp.Maximize'
        code_obj = f"self.optz={sense}({code_obj})"
        if not no_code:
            logger.info(f"Code: {code_obj}")
        self.code = code_obj
        return True

    def evaluate(self):
        """
        Evaluate the objective function.

        Returns
        -------
        bool
            Returns True if the evaluation is successful, False otherwise.
        """
        exec(self.code, globals(), locals())
        return True

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

    def parse(self, no_code=True, force_generate=False):
        """
        Parse the optimization model from the symbolic description.
        Must be called after generating the symbols `self.rtn.syms.generate_symbols()`.

        Parameters
        ----------
        no_code : bool, optional
            Flag indicating if the parsing code should be displayed,
            True by default.
        force_generate : bool, optional
            Flag indicating if the symbols should be generated, goes to `self.rtn.syms.generate_symbols()`.

        Returns
        -------
        bool
            Returns True if the parsing is successful, False otherwise.
        """
        t, _ = elapsed()
        # --- add RParams and Services as parameters ---
        logger.debug(f'Parsing OModel for <{self.rtn.class_name}>')
        for key, val in self.rtn.params.items():
            if not val.no_parse:
                logger.debug(f"    - Param <{key}>")
                try:
                    val.parse()
                except Exception as e:
                    msg = f"Failed to parse Param <{key}>. "
                    msg += f"Original error: {e}"
                    raise Exception(msg)

        # --- add decision variables ---
        for key, val in self.rtn.vars.items():
            try:
                logger.debug(f"    - Var <{key}>")
                val.parse()
            except Exception as e:
                msg = f"Failed to parse Var <{key}>. "
                msg += f"Original error: {e}"
                raise Exception(msg)

        # --- add constraints ---
        for key, val in self.rtn.constrs.items():
            logger.debug(f"    - Constr <{key}>: {val.e_str}")
            try:
                val.parse(no_code=no_code)
            except Exception as e:
                msg = f"Failed to parse Constr <{key}>. "
                msg += f"Original error: {e}"
                raise Exception(msg)

        # --- parse objective functions ---
        if self.rtn.type != 'PF':
            logger.debug(f"    - Objective <{self.rtn.obj.name}>: {self.rtn.obj.e_str}")
            if self.rtn.obj is not None:
                try:
                    self.rtn.obj.parse(no_code=no_code)
                except Exception as e:
                    msg = f"Failed to parse Objective <{self.rtn.obj.name}>. "
                    msg += f"Original error: {e}"
                    raise Exception(msg)
            else:
                logger.warning(f"{self.rtn.class_name} has no objective function!")
                self.parsed = False
                return self.parsed

        # --- parse expressions ---
        for key, val in self.rtn.exprs.items():
            msg = f"    - ExpressionCalc <{key}>: {val.e_str} "
            logger.debug(msg)
            try:
                val.parse(no_code=no_code)
            except Exception as e:
                msg = f"Failed to parse ExpressionCalc <{key}>. "
                msg += f"Original error: {e}"
                raise Exception(msg)
        _, s = elapsed(t)
        logger.debug(f"  -> Parsed in {s}")

        self.parsed = True
        return self.parsed

    def evaluate(self):
        """
        Evaluate the optimization model.
        """
        logger.debug(f"Evaluating OModel for <{self.rtn.class_name}>")
        t, _ = elapsed()
        if not self.parsed:
            raise ValueError("Model is not parsed yet.")
        for key, val in self.rtn.params.items():
            val.evaluate()
            setattr(self, key, val.optz)
        for key, val in self.rtn.vars.items():
            val.evaluate()
            setattr(self, key, val.optz)
        for key, val in self.rtn.constrs.items():
            val.evaluate()
            setattr(self, key, val.optz)
        if self.rtn.type != 'PF':
            self.rtn.obj.evaluate()
            # NOTE: since we already have the attribute `obj`,
            # we can update it rather than setting it
            self.obj = self.rtn.obj.optz
        for key, val in self.rtn.exprs.items():
            val.evaluate()

        self.evaluated = True
        _, s = elapsed(t)
        logger.debug(f" -> Evaluated in {s}")
        return self.evaluated

    def init(self, no_code=True, force_parse=False, force_generate=False):
        """
        Set up the optimization model from the symbolic description.

        This method initializes the optimization model by parsing decision variables,
        constraints, and the objective function from the associated routine.

        Parameters
        ----------
        no_code : bool, optional
            Flag indicating if the parsing code should be displayed,
            True by default.
        force_parse : bool, optional
            Flag indicating if the parsing should be forced, goes to `self.parse()`.
        force_generate : bool, optional
            Flag indicating if the symbols should be generated, goes to `self.parse()`.

        Returns
        -------
        bool
            Returns True if the setup is successful, False otherwise.
        """
        t_init, _ = elapsed()

        if force_parse or not self.parsed:
            self.parse(no_code=no_code, force_generate=force_generate)

        if self.rtn.type == 'PF':
            _, s_init = elapsed(t_init)
            self.initialized = True
            logger.debug(f"OModel for <{self.rtn.class_name}> initialized in {s_init}.")
            return self.initialized

        self.evaluate()

        # --- evaluate the optimziation ---
        t_eva, _ = elapsed()
        code_prob = "self.prob = problem(self.obj, "
        constrs_skip = []
        constrs_add = []
        for key, val in self.rtn.constrs.items():
            if (val.is_disabled) or (val is None):
                constrs_skip.append(f'<{key}>')
            else:
                constrs_add.append(val.optz)
        code_prob += "[constr for constr in constrs_add])"
        for pattern, replacement in self.rtn.syms.sub_map.items():
            code_prob = re.sub(pattern, replacement, code_prob)

        exec(code_prob, globals(), locals())
        _, s_eva = elapsed(t_eva)
        logger.debug(f"OModel for <{self.rtn.class_name}> evaluated in {s_eva}")

        _, s_init = elapsed(t_init)
        self.initialized = True
        logger.debug(f"OModel for <{self.rtn.class_name}> initialized in {s_init}.")

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

    def __setattr__(self, __name: str, __value: Any):
        self._register_attribute(__name, __value)
        super().__setattr__(__name, __value)

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
