import contextlib
import io
import os
import sys
import unittest
from unittest import mock

import ams
import ams.main
import ams.cli


class TestCLI(unittest.TestCase):

    def test_cli_parser(self):
        ams.cli.create_parser()

    def test_cli_version_flag(self):
        parser = ams.cli.create_parser()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with self.assertRaises(SystemExit) as cm:
                parser.parse_args(['--version'])
        self.assertEqual(cm.exception.code, 0)
        out = buf.getvalue().strip()
        self.assertEqual(out, f'ams {ams.__version__}')

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

    def test_selftest_missing_tests_dir(self):
        """
        ``ams st`` must degrade gracefully when the external ``tests/``
        directory is absent — the wheel excludes it from packaging, so
        a wheel install would otherwise crash inside unittest discovery.
        """
        stdout_before = sys.stdout
        with mock.patch.object(ams.main, 'tests_root',
                               return_value='/nonexistent/ams/tests'):
            with self.assertLogs('ams.main', level='WARNING') as cm:
                self.assertIsNone(ams.main.selftest())
        self.assertTrue(any('not packaged with wheel' in m for m in cm.output),
                        msg=f'unexpected log output: {cm.output}')
        # stdout must not be left redirected on the graceful-skip path.
        self.assertIs(sys.stdout, stdout_before)

    def test_profile_run(self):
        _ = ams.main.run(ams.get_case('matpower/case5.m'),
                         no_output=False,
                         profile=True,)
        self.assertTrue(os.path.exists('case5_prof.prof'))
        self.assertTrue(os.path.exists('case5_prof.txt'))
        os.remove('case5_prof.prof')
        os.remove('case5_prof.txt')
