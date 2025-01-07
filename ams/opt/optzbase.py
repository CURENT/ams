"""
Module for optimization base classes.
"""
import logging

from typing import Optional

from ams.utils.misc import deprec_get_idx


logger = logging.getLogger(__name__)


def ensure_symbols(func):
    """
    Decorator to ensure that symbols are generated before parsing.
    If not, it runs self.rtn.syms.generate_symbols().

    Designed to be used on the `parse` method of the optimization elements (`OptzBase`)
    and optimization model (`OModel`), i.e., `Var`, `Param`, `Constraint`, `Objective`,
    and `ExpressionCalc`.

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
    Decorator to ensure that system matrices are built and the OModel is parsed
    before evaluation. If not, it runs the necessary methods to initialize them.

    Designed to be used on the `evaluate` method of optimization elements (`OptzBase`)
    and the optimization model (`OModel`), i.e., `Var`, `Param`, `Constraint`, `Objective`,
    and `ExpressionCalc`.

    Evaluation before building matrices and parsing the OModel can lead to errors. Ensure that
    system matrices are built and the OModel is parsed before calling the `evaluate` method.
    """

    def wrapper(self, *args, **kwargs):
        try:
            if not self.rtn.system.mats.initialized:
                logger.debug("System matrices are not built yet. Building now...")
                self.rtn.system.mats.build()
            if isinstance(self, (OptzBase)):
                if not self.om.parsed:
                    logger.debug("OModel is not parsed yet. Parsing now...")
                    self.om.parse()
            else:
                if not self.parsed:
                    logger.debug("OModel is not parsed yet. Parsing now...")
                    self.parse()
        except Exception as e:
            logger.error(f"Error during initialization or parsing: {e}")
            raise e
        return func(self, *args, **kwargs)
    return wrapper


class OptzBase:
    """
    Base class for optimization elements.
    Ensure that symbols are generated before calling the `parse` method. Parsing
    before symbol generation can lead to incorrect results.

    Parameters
    ----------
    name : str, optional
        Name of the optimization element.
    info : str, optional
        Descriptive information about the optimization element.
    unit : str, optional
        Unit of measurement for the optimization element.

    Attributes
    ----------
    rtn : ams.routines.Routine
        The owner routine instance.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 model: Optional[str] = None,
                 ):
        self.om = None
        self.name = name
        self.info = info
        self.unit = unit
        self.is_disabled = False
        self.rtn = None
        self.optz = None  # corresponding optimization element
        self.code = None
        self.model = model  # indicate if this element belongs to a model or group
        self.owner = None  # instance of the owner model or group
        self.is_group = False

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

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.name}'

    @deprec_get_idx
    def get_idx(self):
        if self.is_group:
            return self.owner.get_all_idxes()
        elif self.owner is None:
            logger.info(f'{self.class_name} <{self.name}> has no owner.')
            return None
        else:
            return self.owner.idx.v

    def get_all_idxes(self):
        """
        Return all the indexes of this item.

        .. note::
            New in version 1.0.0.

        Returns
        -------
        list
            A list of indexes.
        """

        if self.is_group:
            return self.owner.get_all_idxes()
        elif self.owner is None:
            logger.info(f'{self.class_name} <{self.name}> has no owner.')
            return None
        else:
            return self.owner.idx.v
