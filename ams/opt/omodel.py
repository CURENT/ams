"""
Module for optimization modeling.
"""
import logging

from typing import Any, Optional, Union
from collections import OrderedDict
import re

import numpy as np
import scipy.sparse as spr
from scipy.sparse import csr_matrix as c_sparse  # NOQA

from andes.core.common import Config
from andes.utils.misc import elapsed

import cvxpy as cp

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

    def parse(self):
        """
        Parse the object.
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
    complex: bool, optional
        True to set the parameter as complex.
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
                 complex: Optional[bool] = False,
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
                                     ('complex', complex),
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
        """
        config = self.config.as_dict()  # NOQA
        sub_map = self.om.rtn.syms.sub_map
        shape = np.shape(self.v)
        # NOTE: it seems that there is no need to use re.sub here
        code_param = f"self.optz=param(shape={shape}, **config)"
        for pattern, replacement, in sub_map.items():
            code_param = re.sub(pattern, replacement, code_param)
        exec(code_param, globals(), locals())
        try:
            msg = f"Parameter <{self.name}> is set as sparse, "
            msg += "but the value is not sparse."
            val = "self.v"
            if self.sparse:
                if not spr.issparse(self.v):
                    val = "c_sparse(self.v)"
            exec(f"self.optz.value = {val}", globals(), locals())
        except ValueError:
            msg = f"Parameter <{self.name}> has non-numeric value, "
            msg += "no_parse=True is applied."
            logger.warning(msg)
            self.no_parse = True
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
    complex : bool, optional
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
                 complex: Optional[bool] = False,
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
                                     ('complex', complex),
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
        # FIXME: is this safe?
        self._v = value

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
        # only used for CVXPY
        # NOTE: Config only allow lower case letters, do a conversion here
        config = {}
        for k, v in self.config.as_dict().items():
            if k == 'psd':
                config['PSD'] = v
            elif k == 'nsd':
                config['NSD'] = v
            elif k == 'bool':
                config['boolean'] = v
            else:
                config[k] = v
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
        # build the Var object
        exec(code_var, globals(), locals())
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
    type : str, optional
        The type of constraint, which determines the mathematical relationship.
        Possible values include 'uq' (inequality, default) and 'eq' (equality).

    Attributes
    ----------
    is_disabled : bool
        Flag indicating if the constraint is disabled, False by default.
    rtn : ams.routines.Routine
        The owner routine instance.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 e_str: Optional[str] = None,
                 info: Optional[str] = None,
                 type: Optional[str] = 'uq',
                 ):
        OptzBase.__init__(self, name=name, info=info)
        self.e_str = e_str
        self.type = type
        self.is_disabled = False
        self.dual = None
        self.code = None
        # TODO: add constraint info from solver, maybe dual?

    def parse(self, no_code=True):
        """
        Parse the constraint.

        Parameters
        ----------
        no_code : bool, optional
            Flag indicating if the code should be shown, True by default.
        """
        if self.is_disabled:
            return True
        # parse the expression str
        sub_map = self.om.rtn.syms.sub_map
        code_constr = self.e_str
        for pattern, replacement in sub_map.items():
            try:
                code_constr = re.sub(pattern, replacement, code_constr)
            except TypeError as e:
                logger.error(f"Error in parsing constr <{self.name}>.")
                raise e
        # store the parsed expression str code
        self.code = code_constr
        # parse the constraint type
        code_constr = "self.optz=" + code_constr
        if self.type not in ['uq', 'eq']:
            raise ValueError(f'Constraint type {self.type} is not supported.')
        code_constr += " <= 0" if self.type == 'uq' else " == 0"
        msg = f"Parse Constr <{self.name}>: {self.e_str} "
        msg += " <= 0" if self.type == 'uq' else " == 0"
        logger.debug(msg)
        if not no_code:
            logger.info(f"<{self.name}> code: {code_constr}")
        # set the parsed constraint
        exec(code_constr, globals(), locals())
        return True

    def __repr__(self):
        enabled = 'OFF' if self.is_disabled else 'ON'
        out = f"{self.class_name}: {self.name} [{enabled}]"
        return out

    @property
    def v2(self):
        """
        Return the calculated constraint LHS value.
        Note that ``v`` should be used primarily as it is obtained
        from the solver directly.
        ``v2`` is for debugging purpose, and should be consistent with ``v``.
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
            out = eval(code)
            return out
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
        self._v = 0
        self.code = None

    @property
    def v2(self):
        """
        Return the calculated objective value.
        Note that ``v`` should be used primarily as it is obtained
        from the solver directly.
        ``v2`` is for debugging purpose, and should be consistent with ``v``.
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
            out = eval(code)
            return out
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
        out = self.om.obj.value
        out = self._v if out is None else out
        return out

    @v.setter
    def v(self, value):
        self._v = value

    def parse(self, no_code=True):
        """
        Parse the objective function.

        Parameters
        ----------
        no_code : bool, optional
            Flag indicating if the code should be shown, True by default.
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
        logger.debug(f"Parse Objective <{self.name}>: {self.sense.upper()}. {self.e_str}")
        if not no_code:
            logger.info(f"Code: {code_obj}")
        # set the parsed objective function
        exec(code_obj, globals(), locals())
        exec("self.om.obj = self.optz", globals(), locals())
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
    """

    def __init__(self, routine):
        self.rtn = routine
        self.prob = None
        self.params = OrderedDict()
        self.vars = OrderedDict()
        self.constrs = OrderedDict()
        self.obj = None
        self.initialized = False
        self._parsed = False

    def _parse(self, no_code=True):
        """
        Parse the optimization model from the symbolic description.

        Parameters
        ----------
        no_code : bool, optional
            Flag indicating if the parsing code should be displayed,
            True by default.
        """
        rtn = self.rtn
        rtn.syms.generate_symbols(force_generate=False)

        # --- add RParams and Services as parameters ---
        t0, _ = elapsed()
        for key, val in rtn.params.items():
            if not val.no_parse:
                try:
                    val.parse()
                except Exception as e:
                    msg = f"Failed to parse Param <{key}>. "
                    msg += f"Original error: {e}"
                    raise Exception(msg)
                setattr(self, key, val.optz)
        _, s = elapsed(t0)
        logger.debug(f"Parse Params in {s}")

        # --- add decision variables ---
        t0, _ = elapsed()
        for key, val in rtn.vars.items():
            try:
                val.parse()
            except Exception as e:
                msg = f"Failed to parse Var <{key}>. "
                msg += f"Original error: {e}"
                raise Exception(msg)
            setattr(self, key, val.optz)
        _, s = elapsed(t0)
        logger.debug(f"Parse Vars in {s}")

        # --- add constraints ---
        t0, _ = elapsed()
        for key, val in rtn.constrs.items():
            try:
                val.parse(no_code=no_code)
            except Exception as e:
                msg = f"Failed to parse Constr <{key}>. "
                msg += f"Original error: {e}"
                raise Exception(msg)
            setattr(self, key, val.optz)
        _, s = elapsed(t0)
        logger.debug(f"Parse Constrs in {s}")

        # --- parse objective functions ---
        t0, _ = elapsed()
        if rtn.type != 'PF':
            if rtn.obj is not None:
                try:
                    rtn.obj.parse(no_code=no_code)
                except Exception as e:
                    msg = f"Failed to parse Objective <{rtn.obj.name}>. "
                    msg += f"Original error: {e}"
                    raise Exception(msg)
            else:
                logger.warning(f"{rtn.class_name} has no objective function!")
                _, s = elapsed(t0)
                self._parsed = False
                return self._parsed
        _, s = elapsed(t0)
        logger.debug(f"Parse Objective in {s}")

        self._parsed = True
        return self._parsed

    def init(self, no_code=True):
        """
        Set up the optimization model from the symbolic description.

        This method initializes the optimization model by parsing decision variables,
        constraints, and the objective function from the associated routine.

        Parameters
        ----------
        no_code : bool, optional
            Flag indicating if the parsing code should be displayed,
            True by default.

        Returns
        -------
        bool
            Returns True if the setup is successful, False otherwise.
        """
        t_setup, _ = elapsed()

        self._parse(no_code=no_code)

        if self.rtn.type != 'PF':
            # --- finalize the optimziation formulation ---
            code_prob = "self.prob = problem(self.obj, "
            constrs_skip = []
            constrs_add = []
            for key, val in self.rtn.constrs.items():
                if (val is None) or (val.is_disabled):
                    constrs_skip.append(f'<{key}>')
                else:
                    constrs_add.append(val.optz)
            code_prob += "[constr for constr in constrs_add])"
            for pattern, replacement in self.rtn.syms.sub_map.items():
                code_prob = re.sub(pattern, replacement, code_prob)
            msg = f"Finalize: {code_prob}"
            if len(constrs_skip) > 0:
                msg += "; Skipped constrs: "
                msg += ", ".join(constrs_skip)
            logger.debug(msg)
            exec(code_prob, globals(), locals())

        _, s_setup = elapsed(t_setup)
        self.initialized = True
        logger.debug(f"OModel for <{self.rtn.class_name}> initialized in {s_setup}.")

        return self.initialized

    @property
    def class_name(self):
        """
        Return the class name
        """
        return self.__class__.__name__

    def __register_attribute(self, key, value):
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
        self.__register_attribute(__name, __value)
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
