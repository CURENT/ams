"""
Test system report.
"""
import unittest
import os

import ams


import logging

logger = logging.getLogger(__name__)


class TestReport(unittest.TestCase):
    """
    Tests for Report class.
    """

    def setUp(self) -> None:
        self.ss = ams.main.load(
            ams.get_case("5bus/pjm5bus_demo.xlsx"),
            default_config=True,
            no_output=True,
        )
        self.expected_report = 'pjm5bus_demo_out.txt'

    def test_no_output(self):
        """
        Test no output.
        """
        self.assertTrue(self.ss.files.no_output)
        self.assertFalse(self.ss.report())

    def test_no_report(self):
        """
        Test report with no solved routine.
        """
        self.ss.files.no_output = False
        self.assertTrue(self.ss.report())
        self.assertTrue(os.path.exists(self.expected_report))

        with open(self.expected_report, "r") as report_file:
            file_contents = report_file.read()
            self.assertNotIn("DCOPF", file_contents)

        os.remove(self.expected_report)

    def test_DCOPF_report(self):
        """
        Test report with DCOPF solved.
        """
        self.ss.files.no_output = False
        self.ss.DCOPF.run(solver='CLARABEL')
        self.assertTrue(self.ss.report())
        self.assertTrue(os.path.exists(self.expected_report))

        with open(self.expected_report, "r") as report_file:
            file_contents = report_file.read()
            self.assertIn("DCOPF", file_contents)
        os.remove(self.expected_report)

        self.ss.DCOPF.export_csv('./DCOPF.csv')
        self.assertTrue(os.path.exists('./DCOPF.csv'))
        os.remove('./DCOPF.csv')

    def test_DCPF_report(self):
        """
        Test report with DCPF solved.
        """
        self.ss.files.no_output = False
        self.ss.DCPF.run(solver='CLARABEL')
        self.assertTrue(self.ss.report())
        self.assertTrue(os.path.exists(self.expected_report))

        with open(self.expected_report, "r") as report_file:
            file_contents = report_file.read()
            self.assertIn("DCPF", file_contents)
        os.remove(self.expected_report)

        self.ss.DCPF.export_csv('./DCPF.csv')
        self.assertTrue(os.path.exists('./DCPF.csv'))
        os.remove('./DCPF.csv')

    def test_RTED_report(self):
        """
        Test report with RTED solved.
        """
        self.ss.files.no_output = False
        self.ss.RTED.run(solver='CLARABEL')
        self.assertTrue(self.ss.report())
        self.assertTrue(os.path.exists(self.expected_report))

        with open(self.expected_report, "r") as report_file:
            file_contents = report_file.read()
            self.assertIn("RTED", file_contents)
        os.remove(self.expected_report)

        self.ss.RTED.export_csv('./RTED.csv')
        self.assertTrue(os.path.exists('./RTED.csv'))
        os.remove('./RTED.csv')

    def test_RTEDDG_report(self):
        """
        Test report with RTEDDG solved.
        """
        self.ss.files.no_output = False
        self.ss.RTEDDG.run(solver='CLARABEL')
        self.assertTrue(self.ss.report())
        self.assertTrue(os.path.exists(self.expected_report))

        with open(self.expected_report, "r") as report_file:
            file_contents = report_file.read()
            self.assertIn("RTEDDG", file_contents)
        os.remove(self.expected_report)

        self.ss.RTEDDG.export_csv('./RTEDDG.csv')
        self.assertTrue(os.path.exists('./RTEDDG.csv'))
        os.remove('./RTEDDG.csv')

    def test_RTEDES_report(self):
        """
        Test report with RTEDES solved.
        """
        self.ss.files.no_output = False
        self.ss.RTEDES.run(solver='SCIP')
        self.assertTrue(self.ss.report())
        self.assertTrue(os.path.exists(self.expected_report))

        with open(self.expected_report, "r") as report_file:
            file_contents = report_file.read()
            self.assertIn("RTEDES", file_contents)
        os.remove(self.expected_report)

        self.ss.RTEDES.export_csv('./RTEDES.csv')
        self.assertTrue(os.path.exists('./RTEDES.csv'))
        os.remove('./RTEDES.csv')

    def test_ED_report(self):
        """
        Test report with ED solved.
        """
        self.ss.files.no_output = False
        self.ss.ED.run(solver='CLARABEL')
        self.assertTrue(self.ss.report())
        self.assertTrue(os.path.exists(self.expected_report))

        with open(self.expected_report, "r") as report_file:
            file_contents = report_file.read()
            self.assertIn("ED", file_contents)
        os.remove(self.expected_report)

        self.ss.ED.export_csv('./ED.csv')
        self.assertTrue(os.path.exists('./ED.csv'))
        os.remove('./ED.csv')

    def test_EDDG_report(self):
        """
        Test report with EDDG solved.
        """
        self.ss.files.no_output = False
        self.ss.EDDG.run(solver='CLARABEL')
        self.assertTrue(self.ss.report())
        self.assertTrue(os.path.exists(self.expected_report))

        with open(self.expected_report, "r") as report_file:
            file_contents = report_file.read()
            self.assertIn("EDDG", file_contents)
        os.remove(self.expected_report)

        self.ss.EDDG.export_csv('./EDDG.csv')
        self.assertTrue(os.path.exists('./EDDG.csv'))
        os.remove('./EDDG.csv')

    def test_EDES_report(self):
        """
        Test report with EDES solved.
        """
        self.ss.files.no_output = False
        self.ss.EDES.run(solver='SCIP')
        self.assertTrue(self.ss.report())
        self.assertTrue(os.path.exists(self.expected_report))

        with open(self.expected_report, "r") as report_file:
            file_contents = report_file.read()
            self.assertIn("EDES", file_contents)
        os.remove(self.expected_report)

        self.ss.EDES.export_csv('./EDES.csv')
        self.assertTrue(os.path.exists('./EDES.csv'))
        os.remove('./EDES.csv')

    def test_UC_report(self):
        """
        Test report with UC solved.
        """
        self.ss.files.no_output = False
        self.ss.UC.run(solver='SCIP')
        self.assertTrue(self.ss.report())
        self.assertTrue(os.path.exists(self.expected_report))

        with open(self.expected_report, "r") as report_file:
            file_contents = report_file.read()
            self.assertIn("UC", file_contents)
        os.remove(self.expected_report)

        self.ss.UC.export_csv('./UC.csv')
        self.assertTrue(os.path.exists('./UC.csv'))
        os.remove('./UC.csv')

    def test_UCDG_report(self):
        """
        Test report with UCDG solved.
        """
        self.ss.files.no_output = False
        self.ss.UCDG.run(solver='SCIP')
        self.assertTrue(self.ss.report())
        self.assertTrue(os.path.exists(self.expected_report))

        with open(self.expected_report, "r") as report_file:
            file_contents = report_file.read()
            self.assertIn("UCDG", file_contents)
        os.remove(self.expected_report)

        self.ss.UCDG.export_csv('./UCDG.csv')
        self.assertTrue(os.path.exists('./UCDG.csv'))
        os.remove('./UCDG.csv')

    def test_UCES_report(self):
        """
        Test report with UCES solved.
        """
        self.ss.files.no_output = False
        self.ss.UCES.run(solver='SCIP')
        self.assertTrue(self.ss.report())
        self.assertTrue(os.path.exists(self.expected_report))

        with open(self.expected_report, "r") as report_file:
            file_contents = report_file.read()
            self.assertIn("UCES", file_contents)
        os.remove(self.expected_report)

        self.ss.UCES.export_csv('./UCES.csv')
        self.assertTrue(os.path.exists('./UCES.csv'))
        os.remove('./UCES.csv')
