"""
Documenter class for AMS models.
"""
import inspect

import logging
from collections import OrderedDict
from andes.core.documenter import Documenter as andes_Documenter
from andes.utils.tab import make_doc_table, math_wrap

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
        # TODO: fix and add the config doc later on
        # out += self.config.doc(max_width=max_width, export=export)

        return out


class RDocumenter:
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
        self.rparams = parent.rparams
        self.vars = parent.vars
        self.constrs = parent.constrs
        self.obj = parent.obj

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
            out += model_header + f'Routine <{self.class_name}> in Type <{self.parent.type}>\n' + model_header

        if self.parent.__doc__ is not None:
            out += inspect.cleandoc(self.parent.__doc__)
        out += '\n\n'  # this fixes the indentation for the next line

        # add tables
        # TODO: fix obj and constr doc
        # out += self._obj_doc(max_width=max_width, export=export)
        # out += self._constr_doc(max_width=max_width, export=export)
        out += self._var_doc(max_width=max_width, export=export)
        out += self._param_doc(max_width=max_width, export=export)

        return out

    def _constr_doc(self, max_width=78, export='plain'):
        # constraint documentation
        if len(self.constrs) == 0:
            return ''

        # prepare temporary lists
        names, class_names, info = list(), list(), list()

        for p in self.constrs.values():
            names.append(p.name)
            class_names.append(p.class_name)
            info.append(p.info if p.info else '')

        # symbols based on output format
        if export == 'rest':
            symbols = [item.tex_name for item in self.constrs.values()]
            symbols = math_wrap(symbols, export=export)
            title = 'Constraints\n----------------------------------'
        else:
            symbols = [item.name for item in self.constrs.values()]
            title = 'Constraints'

        plain_dict = OrderedDict([('Name', names),
                                  ('Description', info),
                                  ])

        rest_dict = OrderedDict([('Name', names),
                                 ('Symbol', symbols),
                                 ('Description', info),
                                 ])

        # convert to rows and export as table
        return make_doc_table(title=title,
                              max_width=max_width,
                              export=export,
                              plain_dict=plain_dict,
                              rest_dict=rest_dict)

    def _obj_doc(self, max_width=78, export='plain'):
        # variable documentation
        if self.parent.obj is None:
            return ''

        # prepare temporary lists
        names, class_names, info = list(), list(), list()

        p = self.parent.obj
        names.append(p.name)
        class_names.append(p.class_name)
        info.append(p.info if p.info else '')

        # symbols based on output format
        if export == 'rest':
            symbols = [item.tex_name for item in self.vars.values()]
            symbols = math_wrap(symbols, export=export)
            title = 'Objective\n----------------------------------'
        else:
            symbols = [item.name for item in self.vars.values()]
            title = 'Objective'
        
        plain_dict = OrderedDict([('Name', names),
                                  ('Description', info),
                                  ])

        rest_dict = OrderedDict([('Name', names),
                                 ('Symbol', symbols),
                                 ('Description', info),
                                 ])

        # convert to rows and export as table
        return make_doc_table(title=title,
                              max_width=max_width,
                              export=export,
                              plain_dict=plain_dict,
                              rest_dict=rest_dict)


    def _var_doc(self, max_width=78, export='plain'):
        # variable documentation
        if len(self.vars) == 0:
            return ''

        # prepare temporary lists
        names, units, class_names = list(), list(), list()
        info = list()
        units_rest = list()

        for p in self.vars.values():
            names.append(p.name)
            class_names.append(p.class_name)
            info.append(p.info if p.info else '')
            # defaults.append(p.default if p.default is not None else '')
            units.append(f'{p.unit}' if p.unit else '')
            units_rest.append(f'*{p.unit}*' if p.unit else '')

            # plist = []
            # for key, val in p.property.items():
            #     if val is True:
            #         plist.append(key)
            # properties.append(','.join(plist))

        # symbols based on output format
        if export == 'rest':
            symbols = [item.tex_name for item in self.vars.values()]
            symbols = math_wrap(symbols, export=export)
            title = 'Routine Algebs\n----------------------------------'
        else:
            symbols = [item.name for item in self.vars.values()]
            title = 'Routine Algebs'

        plain_dict = OrderedDict([('Name', names),
                                  ('Description', info),
                                  ('Unit', units),
                                  ])

        rest_dict = OrderedDict([('Name', names),
                                 ('Symbol', symbols),
                                 ('Description', info),
                                 ('Unit', units_rest),
                                 ])

        # convert to rows and export as table
        return make_doc_table(title=title,
                              max_width=max_width,
                              export=export,
                              plain_dict=plain_dict,
                              rest_dict=rest_dict)

    def _param_doc(self, max_width=78, export='plain'):
        """
        Export formatted routine parameter documentation as a string.

        Parameters
        ----------
        max_width : int, optional = 80
            Maximum table width. If export format is ``rest`` it will be unlimited.

        export : str, optional = 'plain'
            Export format, 'plain' for plain text, 'rest' for restructuredText.

        Returns
        -------
        str
            Tabulated output in a string
        """
        if len(self.rparams) == 0:
            return ''

        # prepare temporary lists
        names, units, class_names = list(), list(), list()
        info = list()
        units_rest = list()

        for p in self.rparams.values():
            names.append(p.name)
            class_names.append(p.class_name)
            info.append(p.info if p.info else '')
            # defaults.append(p.default if p.default is not None else '')
            units.append(f'{p.unit}' if p.unit else '')
            units_rest.append(f'*{p.unit}*' if p.unit else '')

            # plist = []
            # for key, val in p.property.items():
            #     if val is True:
            #         plist.append(key)
            # properties.append(','.join(plist))

        # symbols based on output format
        if export == 'rest':
            symbols = [item.tex_name for item in self.rparams.values()]
            symbols = math_wrap(symbols, export=export)
            title = 'Routine Parameters\n----------------------------------'
        else:
            symbols = [item.name for item in self.rparams.values()]
            title = 'Routine Parameters'

        plain_dict = OrderedDict([('Name', names),
                                  ('Description', info),
                                  ('Unit', units),
                                  ])

        rest_dict = OrderedDict([('Name', names),
                                 ('Symbol', symbols),
                                 ('Description', info),
                                 ('Unit', units_rest),
                                 ])

        # convert to rows and export as table
        return make_doc_table(title=title,
                              max_width=max_width,
                              export=export,
                              plain_dict=plain_dict,
                              rest_dict=rest_dict)
