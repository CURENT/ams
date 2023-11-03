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
