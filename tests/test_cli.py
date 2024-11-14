import unittest

import ams.main


class TestCLI(unittest.TestCase):
    def test_main_doc(self):
        ams.main.doc('Bus')
        ams.main.doc(list_supported=True)

    def test_versioninfo(self):
        ams.main.versioninfo()

    def test_misc(self):
        ams.main.misc(show_license=True)
        ams.main.misc(save_config=None, overwrite=True)
