import logging
import inspect
from collections import OrderedDict

logger = logging.getLogger(__name__)


class TypeBase:
    """
    Base class for types.
    """

    def __init__(self):

        self.common_rparams = []
        self.common_vars = []
        self.common_constrs = []

        self.routines = OrderedDict()

    @property
    def class_name(self):
        return self.__class__.__name__

    @property
    def n(self):
        """
        Total number of routines.
        """
        return len(self.routines)

    def __repr__(self):
        dev_text = 'routine' if self.n == 1 else 'routines'
        return f'{self.class_name} ({self.n} {dev_text}) at {hex(id(self))}'

    def doc(self, export='plain'):
        """
        Return the documentation of the type in a string.
        """
        out = ''
        if export == 'rest':
            out += f'.. _{self.class_name}:\n\n'
            group_header = '=' * 80 + '\n'
        else:
            group_header = ''

        if export == 'rest':
            out += group_header + f'{self.class_name}\n' + group_header
        else:
            out += group_header + f'Type <{self.class_name}>\n' + group_header

        if self.__doc__ is not None:
            out += inspect.cleandoc(self.__doc__) + '\n\n'

        if len(self.common_rparams):
            out += 'Common Parameters: ' + ', '.join(self.common_rparams)
            out += '\n\n'
        if len(self.common_vars):
            out += 'Common Vars: ' + ', '.join(self.common_vars)
            out += '\n\n'
        if len(self.common_constrs):
            out += 'Common Constraints: ' + ', '.join(self.common_constrs)
            out += '\n\n'
        if len(self.routines):
            out += 'Available routines:\n'
            rtn_name_list = list(self.routines.keys())

            if export == 'rest':
                def add_reference(name_list):
                    return [f'{item}_' for item in name_list]

                rtn_name_list = add_reference(rtn_name_list)

            out += ',\n'.join(rtn_name_list) + '\n'

        return out

    def doc_all(self, export='plain'):
        """
        Return documentation of the type and its routines.

        Parameters
        ----------
        export : 'plain' or 'rest'
            Export format, plain-text or RestructuredText

        Returns
        -------
        str

        """
        out = self.doc(export=export)
        out += '\n'
        for instance in self.routines.values():
            out += instance.doc(export=export)
            out += '\n'
        return out


class UndefinedType(TypeBase):
    """
    The undefined type.
    """

    def __init__(self):
        TypeBase.__init__(self)


class PF(TypeBase):
    """
    Type for power flow routines.
    """

    def __init__(self):
        TypeBase.__init__(self)
        self.common_rparams.extend(('pd',))
        self.common_vars.extend(('pg',))


class DCED(TypeBase):
    """
    Type for DC-based economic dispatch.
    """

    def __init__(self):
        TypeBase.__init__(self)
        self.common_rparams.extend(('c2', 'c1', 'c0', 'pmax', 'pmin', 'pd', 'ptdf', 'rate_a',))
        self.common_vars.extend(('pg',))
        self.common_constrs.extend(('pb', 'lub', 'llb'))


class DCUC(TypeBase):
    """
    Type for DC-based unit commitment.
    """

    def __init__(self):
        TypeBase.__init__(self)
        # TODO: add common parameters and variables


class DED(TypeBase):
    """
    Type for Distributional economic dispatch.
    """

    def __init__(self):
        TypeBase.__init__(self)
        # TODO: add common parameters and variables


class ACED(DCED):
    """
    Type for AC-based economic dispatch.
    """

    def __init__(self):
        DCED.__init__(self)
        self.common_rparams.extend(('qd',))
        self.common_vars.extend(('aBus', 'vBus', 'qg',))
