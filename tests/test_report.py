"""
Test system report.
"""
import unittest
import os

import ams


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
        self.ss.DCOPF.run(solver='ECOS')
        self.assertTrue(self.ss.report())
        self.assertTrue(os.path.exists(self.expected_report))

        with open(self.expected_report, "r") as report_file:
            file_contents = report_file.read()
            self.assertIn("DCOPF", file_contents)

        os.remove(self.expected_report)

    def test_multi_report(self):
        """
        Test report with multiple solved routines.
        """
        self.ss.files.no_output = False
        self.ss.DCOPF.run(solver='ECOS')
        self.ss.RTED.run(solver='ECOS')
        self.ss.ED.run(solver='ECOS')
        self.assertTrue(self.ss.report())
        self.assertTrue(os.path.exists(self.expected_report))

        with open(self.expected_report, "r") as report_file:
            file_contents = report_file.read()
            self.assertIn("DCOPF", file_contents)
            self.assertIn("RTED", file_contents)
            self.assertIn("ED", file_contents)

        os.remove(self.expected_report)
