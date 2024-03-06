import unittest

import ams


class TestCodegen(unittest.TestCase):
    """
    Test code generation.
    """

    def test_1_docs(self) -> None:
        sp = ams.system.System()
        out = ''
        for tp in sp.types.values():
            out += tp.doc_all()

        out = ''
        for group in sp.groups.values():
            out += group.doc_all()

    def test_docum(self) -> None:
        sp = ams.load(ams.get_case('5bus/pjm5bus_demo.xlsx'),
                      setup=True, no_output=True)
        sp.DCOPF.init()
        docum = sp.DCOPF.docum
        for export in ['plain', 'rest']:
            docum._obj_doc(max_width=78, export=export)
            docum._constr_doc(max_width=78, export=export)
            docum._expr_doc(max_width=78, export=export)
            docum._var_doc(max_width=78, export=export)
            docum._service_doc(max_width=78, export=export)
            docum._param_doc(max_width=78, export=export)
            docum.config.doc(max_width=78, export=export)
