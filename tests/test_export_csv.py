"""
Test routine export to CSV.
"""
import unittest
import os
import csv

import numpy as np

import ams


class TestExportCSV(unittest.TestCase):
    """
    Tests for Routine export to CSV.
    """

    def setUp(self) -> None:
        self.ss = ams.main.load(
            ams.get_case("5bus/pjm5bus_demo.xlsx"),
            default_config=True,
            no_output=True,
        )
        self.expected_csv_DCOPF = 'pjm5bus_demo_DCOPF.csv'
        self.expected_csv_ED = 'pjm5bus_demo_ED.csv'

    def test_no_export(self):
        """
        Test no export when routine is not converged.
        """
        self.assertIsNone(self.ss.DCOPF.export_csv())

    def test_export_DCOPF(self):
        """
        Test export DCOPF to CSV.
        """
        self.ss.DCOPF.run(solver='ECOS')
        self.assertTrue(self.ss.DCOPF.export_csv())
        self.assertTrue(os.path.exists(self.expected_csv_DCOPF))

        n_rows = 0
        n_cols = 0
        with open(self.expected_csv_DCOPF, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                n_rows += 1
                # Check if this row has more columns than the previous rows
                if n_cols == 0 or len(row) > n_cols:
                    n_cols = len(row)

        n_cols_expected = np.sum([v.shape[0] for v in self.ss.DCOPF.vars.values()])
        # cols number plus one for the index column
        self.assertEqual(n_cols, n_cols_expected + 1)
        # header row plus data row
        n_rows_expected = 2
        self.assertEqual(n_rows, n_rows_expected)

        os.remove(self.expected_csv_DCOPF)

    def test_export_ED(self):
        """
        Test export ED to CSV.
        """
        self.ss.ED.run(solver='ECOS')
        self.assertTrue(self.ss.ED.export_csv())
        self.assertTrue(os.path.exists(self.expected_csv_ED))

        n_rows = 0
        n_cols = 0
        with open(self.expected_csv_ED, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                n_rows += 1
                # Check if this row has more columns than the previous rows
                if n_cols == 0 or len(row) > n_cols:
                    n_cols = len(row)

        n_cols_expected = np.sum([v.shape[0] for v in self.ss.ED.vars.values()])
        # cols number plus one for the index column
        self.assertEqual(n_cols, n_cols_expected + 1)
        # header row plus data row
        n_rows_expected = len(self.ss.ED.timeslot.v) + 1
        self.assertTrue(n_rows, n_rows_expected)

        os.remove(self.expected_csv_ED)
