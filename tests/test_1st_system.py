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
        sp = ams.load(ams.get_case('5bus/pjm5bus_demo.json'),
                      setup=True, no_output=True)
        sp.DCOPF.init()
        docum = sp.DCOPF.docum
        for export in ['plain', 'rest']:
            docum._obj_doc(max_width=78, export=export)
            docum._constr_doc(max_width=78, export=export)
            docum._exprc_doc(max_width=78, export=export)
            docum._var_doc(max_width=78, export=export)
            docum._service_doc(max_width=78, export=export)
            docum._param_doc(max_width=78, export=export)
            docum.parent.config.doc(max_width=78, export=export)


class TestParamCorrection(unittest.TestCase):
    """
    Test parameter correction.
    """

    def setUp(self) -> None:
        """
        Test setup.
        """
        self.ss = ams.load(ams.get_case('matpower/case14.m'),
                           setup=False, no_output=True, default_config=True,)

    def test_line_correction(self):
        """
        Test line correction.
        """
        self.ss.Line.rate_a.v[5] = 0.0
        self.ss.Line.rate_b.v[6] = 0.0
        self.ss.Line.rate_c.v[7] = 0.0
        self.ss.Line.amax.v[8] = 0.0
        self.ss.Line.amin.v[9] = 0.0

        self.ss.setup()

        self.assertIsNot(self.ss.Line.rate_a.v[5], 0.0)
        self.assertIsNot(self.ss.Line.rate_b.v[6], 0.0)
        self.assertIsNot(self.ss.Line.rate_c.v[7], 0.0)
        self.assertIsNot(self.ss.Line.amax.v[8], 0.0)
        self.assertIsNot(self.ss.Line.amin.v[9], 0.0)
