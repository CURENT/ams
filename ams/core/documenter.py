"""
Documenter class for AMS models.
"""
import inspect
import re

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
        self.algebs = parent.algebs
        self.services = parent.services

        func_to_disable = ['_init_doc', '_eq_doc',
                           '_discrete_doc', '_block_doc']
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
        out += self._var_doc(max_width=max_width, export=export)
        out += self._service_doc(max_width=max_width, export=export)
        # TODO: fix and add the config doc later on
        # out += self.config.doc(max_width=max_width, export=export)

        return out

    def _var_doc(self, max_width=78, export='plain'):
        # variable documentation
        if len(self.algebs) == 0:
            return ''

        names, symbols, units = list(), list(), list()
        info = list()
        units_rest, ty = list(), list()

        for p in self.algebs.values():
            names.append(p.name)
            ty.append(p.class_name)
            info.append(p.info if p.info else '')
            units.append(p.unit if p.unit else '')
            units_rest.append(f'*{p.unit}*' if p.unit else '')

        title = 'Variables'

        # replace with latex math expressions if export is ``rest``
        if export == 'rest':
            symbols = [item.tex_name for item in self.algebs.values()]
            symbols = math_wrap(symbols, export=export)
            title = 'Variables\n---------'

        plain_dict = OrderedDict([('Name', names),
                                  ('Type', ty),
                                  ('Description', info),
                                  ('Unit', units)])

        rest_dict = OrderedDict([('Name', names),
                                 ('Symbol', symbols),
                                 ('Type', ty),
                                 ('Description', info),
                                 ('Unit', units_rest)])

        return make_doc_table(title=title,
                              max_width=max_width,
                              export=export,
                              plain_dict=plain_dict,
                              rest_dict=rest_dict)

    def _service_doc(self, max_width=78, export='plain'):
        if len(self.services) == 0:
            return ''

        names, symbols = list(), list()
        info = list()
        class_names = list()

        for p in self.services.values():
            names.append(p.name)
            class_names.append(p.class_name)
            info.append(p.info if p.info else '')
            symbols.append(p.tex_name if p.tex_name is not None else '')

        title = 'Services'
        if export == 'rest':
            symbols = math_wrap(symbols, export=export)
            title = 'Services\n----------'

        plain_dict = OrderedDict([('Name', names),
                                  ('Description', info),
                                  ('Type', class_names)])

        rest_dict = OrderedDict([('Name', names),
                                 ('Description', info),
                                 ('Symbol', symbols),
                                 ('Type', class_names)])

        return make_doc_table(title=title,
                              max_width=max_width,
                              export=export,
                              plain_dict=plain_dict,
                              rest_dict=rest_dict)


class RDocumenter:
    """
    Helper class for documenting routines.

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
        self.services = parent.services
        self.rparams = parent.rparams
        self.vars = parent.vars
        self.constrs = parent.constrs
        self.obj = parent.obj

    def get(self, max_width=78, export='plain'):
        """
        Return the routine documentation in table-formatted string.

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
        self.parent.syms.generate_symbols()
        out += self._obj_doc(max_width=max_width, export=export)
        out += self._constr_doc(max_width=max_width, export=export)
        out += self._var_doc(max_width=max_width, export=export)
        out += self._service_doc(max_width=max_width, export=export)
        out += self._param_doc(max_width=max_width, export=export)
        out += self.config.doc(max_width=max_width, export=export)

        return out

    def _service_doc(self, max_width=78, export='plain'):
        # routine service documentation
        if len(self.services) == 0:
            return ''

        names, symbols = list(), list()
        info = list()
        class_names = list()

        for p in self.services.values():
            names.append(p.name)
            info.append(p.info if p.info else '')
            class_names.append(p.class_name)

        title = 'Services'

        # replace with latex math expressions if export is ``rest``
        if export == 'rest':
            symbols = [item.tex_name for item in self.services.values()]
            symbols = math_wrap(symbols, export=export)
            title = 'Services\n---------'

        plain_dict = OrderedDict([('Name', names),
                                  ('Description', info),
                                  ('Type', class_names)])

        rest_dict = OrderedDict([('Name', names),
                                 ('Symbol', symbols),
                                 ('Description', info),
                                 ('Type', class_names)])

        return make_doc_table(title=title,
                              max_width=max_width,
                              export=export,
                              plain_dict=plain_dict,
                              rest_dict=rest_dict)

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

        # expressions based on output format
        expressions = []
        if export == 'rest':
            for p in self.constrs.values():
                expr = _tex_pre(self, p, self.parent.syms.tex_map)
                if p.type == 'eq':
                    expr = f'{expr} = 0'
                elif p.type == 'uq':
                    expr = f'{expr} \\leq 0'
                logger.debug(f'{p.name} math: {expr}')
                expressions.append(expr)

            title = 'Constraints\n----------------------------------'
        else:
            title = 'Constraints'
        expressions = math_wrap(expressions, export=export)

        plain_dict = OrderedDict([('Name', names),
                                  ('Description', info),
                                  ])

        rest_dict = OrderedDict([('Name', names),
                                 ('Description', info),
                                 ('Expression', expressions),
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
        units, units_rest = list(), list()

        p = self.parent.obj
        units.append(p.unit if p.unit else '')
        units_rest.append(f'*{p.unit}*' if p.unit else '')

        # expressions based on output format
        expr = _tex_pre(self, p, self.parent.syms.tex_map)
        expr = p.sense + '. ' + expr  # minimize or maximize
        expr = [expr]
        if export == 'rest':
            expr = math_wrap(expr, export=export)
            title = 'Objective\n----------------------------------'
        else:
            title = 'Objective'

        plain_dict = OrderedDict([('Unit', units),])

        rest_dict = OrderedDict([('Unit', units_rest),
                                 ('Expression', expr)])

        # convert to rows and export as table
        return make_doc_table(title=title,
                              max_width=max_width,
                              export=export,
                              plain_dict=plain_dict,
                              rest_dict=rest_dict)

    def _var_doc(self, max_width=78, export='plain'):
        # NOTE: this is for the optimization variables
        # not the _var_doc for ANDES parameters
        # variable documentation
        if len(self.vars) == 0:
            return ''

        # prepare temporary lists
        names, units, class_names = list(), list(), list()
        properties, info = list(), list()
        sources = list()
        units_rest = list()

        for p in self.vars.values():
            names.append(p.name)
            class_names.append(p.class_name)
            info.append(p.info if p.info else '')
            # defaults.append(p.default if p.default is not None else '')
            units.append(f'{p.unit}' if p.unit else '')
            units_rest.append(f'*{p.unit}*' if p.unit else '')

            # collect properties defined in config
            plist = []
            for key, val in p.config.as_dict().items():
                if val is True:
                    plist.append(key)
            properties.append(','.join(plist))

            slist = []
            if p.owner is not None and p.src is not None:
                slist.append(f'{p.owner.class_name}.{p.src}')
            sources.append(','.join(slist))

        # symbols based on output format
        if export == 'rest':
            symbols = [item.tex_name for item in self.vars.values()]
            symbols = math_wrap(symbols, export=export)
            title = 'Vars\n----------------------------------'
        else:
            symbols = [item.name for item in self.vars.values()]
            title = 'Vars'

        plain_dict = OrderedDict([('Name', names),
                                  ('Description', info),
                                  ('Unit', units),
                                  ('Properties', properties)])

        rest_dict = OrderedDict([('Name', names),
                                 ('Symbol', symbols),
                                 ('Description', info),
                                 ('Unit', units_rest),
                                 ('Source', sources),
                                 ('Properties', properties)])

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
        sources = list()
        units_rest = list()

        for p in self.rparams.values():
            names.append(p.name)
            class_names.append(p.class_name)
            info.append(p.info if p.info else '')
            # defaults.append(p.default if p.default is not None else '')
            units.append(f'{p.unit}' if p.unit else '')
            units_rest.append(f'*{p.unit}*' if p.unit else '')

            slist = []
            if p.owner is not None and p.src is not None:
                slist.append(f'{p.owner.class_name}.{p.src}')
            sources.append(','.join(slist))

        # symbols based on output format
        if export == 'rest':
            symbols = [item.tex_name for item in self.rparams.values()]
            symbols = math_wrap(symbols, export=export)
            title = 'Parameters\n----------------------------------'
        else:
            symbols = [item.name for item in self.rparams.values()]
            title = 'Parameters'

        plain_dict = OrderedDict([('Name', names),
                                  ('Description', info),
                                  ('Unit', units),
                                  ])

        rest_dict = OrderedDict([('Name', names),
                                 ('Symbol', symbols),
                                 ('Description', info),
                                 ('Unit', units_rest),
                                 ('Source', sources),
                                 ])

        # convert to rows and export as table
        return make_doc_table(title=title,
                              max_width=max_width,
                              export=export,
                              plain_dict=plain_dict,
                              rest_dict=rest_dict)


def _tex_pre(docm, p, tex_map):
    """
    Prepare the expression for pretty printing.

    Parameters
    ----------
    docm : Documenter
        The Documenter instance.
    p : obj or const
        The objective or constraint instance.
    tex_map : dict
        The tex map to use.
    """

    # NOTE: in the future, there might occur special math symbols
    map_before = OrderedDict([
        ('sum', 'SUM'),
        (r'\sum', 'SUM'),
        (r'\eta', 'ETA'),
        (r'\gamma', 'GAMMA'),
        (r'\theta', 'THETA'),
        (r'\frac', 'FRAC'),
        (r'\overline', 'OVERLINE'),
        (r'\underline', 'UNDERLINE'),
    ])
    map_post = OrderedDict([
        ('SUM', r'\sum'),
        ('THETA', r'\theta'),
        ('ETA', r'\eta'),
        ('GAMMA', r'\gamma'),
        ('FRAC', r'\frac'),
        ('OVERLINE', r'\overline'),
        ('UNDERLINE', r'\underline'),
    ])

    expr = p.e_str

    for pattern, replacement in tex_map.items():
        for key, val in map_before.items():
            if key in replacement:
                replacement = replacement.replace(key, val)
        if r'\p' in replacement:
            continue
        try:
            expr = re.sub(pattern, replacement, expr)
        except re.error:
            expr_pattern = pattern.removeprefix('\\b').removesuffix('\\b')
            msg = f'Failed to parse <{expr_pattern}> in {docm.parent.class_name} <{p.name}>, check its tex_name.'
            logger.error(msg)
            expr = ''
        try:
            expr = expr.replace('*', ' ')
        except re.error:
            logger.error('Remains '*' in the expression.')

    for pattern, replacement in map_post.items():
        expr = expr.replace(pattern, replacement)

    return expr
