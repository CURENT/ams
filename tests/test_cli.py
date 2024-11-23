import unittest
import os

import ams.main
import ams.cli


class TestCLI(unittest.TestCase):

    def test_cli_parser(self):
        ams.cli.create_parser()

    def test_cli_preamble(self):
        ams.cli.preamble()

    def test_main_doc(self):
        ams.main.doc('Bus')
        ams.main.doc(list_supported=True)

    def test_versioninfo(self):
        ams.main.versioninfo()

    def test_misc(self):
        ams.main.misc(show_license=True)
        ams.main.misc(save_config=None, overwrite=True)

    def test_profile_run(self):
        _ = ams.main.run(ams.get_case('matpower/case5.m'),
                         no_output=False,
                         profile=True,)
        self.assertTrue(os.path.exists('case5_prof.prof'))
        self.assertTrue(os.path.exists('case5_prof.txt'))
        os.remove('case5_prof.prof')
        os.remove('case5_prof.txt')
