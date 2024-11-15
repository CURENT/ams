import unittest

import ams


class TestOModelWrappers(unittest.TestCase):
    """
    Test wrappers in module `omodel`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("matpower/case5.m"),
                           setup=True,
                           default_config=True,
                           no_output=True,
                           )

    def test_ensure_symbols(self):
        """
        Test `ensure_symbols` wrapper.
        """
        self.assertFalse(self.ss.DCOPF._syms, "Symbols should not be ready before generation!")
        self.ss.DCOPF.pd.parse()
        self.assertTrue(self.ss.DCOPF._syms, "Symbols should be ready after generation!")

    def test_ensure_mats_and_parsed(self):
        """
        Test `ensure_mats_and_parsed` wrapper.
        """
        self.assertFalse(self.ss.mats.initialized, "MatProcessor should not be initialized before initialization!")
        self.assertFalse(self.ss.DCOPF.om.parsed, "OModel should be parsed after parsing!")
        self.ss.DCOPF.pd.evaluate()
        self.assertTrue(self.ss.mats.initialized, "MatProcessor should be initialized after parsing!")
        self.assertTrue(self.ss.DCOPF.om.parsed, "OModle should be parsed after parsing!")


class TestOModel(unittest.TestCase):
    """
    Test class `OModel`.
    """

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case("matpower/case5.m"),
                           setup=True,
                           default_config=True,
                           no_output=True,
                           )

    def test_initialized(self):
        """
        Test `OModel` initialization.
        """
        self.assertFalse(self.ss.DCOPF.om.initialized, "OModel shoud not be initialized before initialziation!")
        self.ss.DCOPF.om.init()
        self.assertTrue(self.ss.DCOPF.om.initialized, "OModel should be initialized!")

    def test_uninitialized_after_nonparametric_update(self):
        """
        Test `OModel` initialization after nonparametric update.
        """
        self.ss.DCOPF.om.init()
        self.assertTrue(self.ss.DCOPF.om.initialized, "OModel should be initialized after initialization!")
        self.ss.DCOPF.update('ug')
        self.assertTrue(self.ss.DCOPF.om.initialized, "OModel should be initialized after nonparametric update!")
