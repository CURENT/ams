"""
Documenter class for AMS models.
"""
import inspect

import logging
from andes.core.documenter import Documenter as andes_Documenter

logger = logging.getLogger(__name__)


def disable_method(func):
    def wrapper(*args, **kwargs):
        msg = f"Method `{func.__name__}` is included in ANDES Documenter but not supported in AMS Documenter."
        logger.warning(msg)
        return None
    return wrapper


def disable_methods(methods):
    for method in methods:
        setattr(Documenter, method, disable_method(getattr(Documenter, method)))


class Documenter(andes_Documenter):
    """
    Helper class for documenting models.

    Parameters
    ----------
    parent : Model
        The `Model` instance to document
    """

    def __init__(self, parent):
        self.parent = parent
        self.system = parent.system
        self.class_name = parent.class_name
        self.config = parent.config
        self.params = parent.params

        func_to_disable = ['_var_doc', '_init_doc', '_eq_doc',
                           '_service_doc', '_discrete_doc', '_block_doc']
        disable_methods(func_to_disable)

    def get(self, max_width=78, export='plain'):
        """
        Return the model documentation in table-formatted string.

        Parameters
        ----------
        max_width : int
            Maximum table width. Automatically et to 0 if format is ``rest``.
        export : str, ('plain', 'rest')
            Export format. Use fancy table if is ``rest``.

        Returns
        -------
        str
            A string with the documentations.
        """
        out = ''
        if export == 'rest':
            max_width = 0
            model_header = '-' * 80 + '\n'
            out += f'.. _{self.class_name}:\n\n'
        else:
            model_header = ''

        if export == 'rest':
            out += model_header + f'{self.class_name}\n' + model_header
        else:
            out += model_header + f'Model <{self.class_name}> in Group <{self.parent.group}>\n' + model_header

        if self.parent.__doc__ is not None:
            out += inspect.cleandoc(self.parent.__doc__)
        out += '\n\n'  # this fixes the indentation for the next line

        # add tables
        out += self._param_doc(max_width=max_width, export=export)
        out += self.config.doc(max_width=max_width, export=export)

        return out
