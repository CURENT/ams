"""
Test routine export to CSV.
"""
import unittest
import os
import csv
import tempfile

import numpy as np

import ams


class TestExportCSV(unittest.TestCase):
    """
    Tests for Routine export to CSV.
    """

    def setUp(self) -> None:
        self.ss = ams.main.load(
            ams.get_case("5bus/pjm5bus_demo.json"),
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
        self.ss.DCOPF.run(solver='CLARABEL')
        self.ss.DCOPF.export_csv()
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

        n_cols_expected = np.sum([v.owner.n for v in self.ss.DCOPF.vars.values()])
        n_cols_expected += np.sum([v.owner.n for v in self.ss.DCOPF.exprs.values()])
        n_cols_expected += np.sum([v.owner.n for v in self.ss.DCOPF.exprcs.values()])
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
        self.ss.ED.run(solver='CLARABEL')
        self.ss.ED.export_csv()
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

        n_cols_expected = np.sum([v.owner.n for v in self.ss.ED.vars.values()])
        n_cols_expected += np.sum([v.owner.n for v in self.ss.ED.exprs.values()])
        n_cols_expected += np.sum([v.owner.n for v in self.ss.ED.exprcs.values()])
        # cols number plus one for the index column
        self.assertEqual(n_cols, n_cols_expected + 1)
        # header row plus data row
        n_rows_expected = len(self.ss.ED.timeslot.v) + 1
        self.assertTrue(n_rows, n_rows_expected)

        os.remove(self.expected_csv_ED)

    def test_load_csv_DCOPF(self):
        """
        Round-trip ``DCOPF.export_csv`` → ``load_csv`` for a
        single-period routine.
        """
        with tempfile.TemporaryDirectory() as tmp:
            self.ss.DCOPF.run(solver='CLARABEL')
            target = os.path.join(tmp, 'dcopf.csv')
            self.ss.DCOPF.export_csv(path=target)

            ss2 = ams.main.load(
                ams.get_case("5bus/pjm5bus_demo.json"),
                default_config=True,
                no_output=True,
            )
            self.assertFalse(ss2.DCOPF.converged)
            self.assertTrue(ss2.DCOPF.load_csv(target))
            self.assertTrue(ss2.DCOPF.converged)
            self.assertEqual(ss2.DCOPF.exit_code, 0)
            np.testing.assert_allclose(ss2.DCOPF.pg.v,
                                       self.ss.DCOPF.pg.v, atol=1e-5)
            self.assertEqual(ss2.DCOPF.pg.v.shape,
                             (self.ss.StaticGen.n,))

    def test_load_csv_ED(self):
        """
        Round-trip ``ED.export_csv`` → ``load_csv`` for a multi-period
        routine. Verifies the ``(n_dev, n_slot)`` reshape.
        """
        with tempfile.TemporaryDirectory() as tmp:
            self.ss.ED.run(solver='CLARABEL')
            target = os.path.join(tmp, 'ed.csv')
            self.ss.ED.export_csv(path=target)

            ss2 = ams.main.load(
                ams.get_case("5bus/pjm5bus_demo.json"),
                default_config=True,
                no_output=True,
            )
            self.assertTrue(ss2.ED.load_csv(target))
            self.assertTrue(ss2.ED.converged)
            self.assertEqual(ss2.ED.pg.v.shape,
                             (ss2.StaticGen.n, ss2.EDTSlot.n))
            np.testing.assert_allclose(ss2.ED.pg.v,
                                       self.ss.ED.pg.v, atol=1e-5)

    def test_export_default_uses_output_path(self):
        """
        With ``output_path`` set, ``export_csv()`` (no path arg) writes
        there instead of CWD.
        """
        with tempfile.TemporaryDirectory() as tmp:
            ss = ams.main.load(
                ams.get_case("5bus/pjm5bus_demo.json"),
                default_config=True,
                no_output=False,
                output_path=tmp,
            )
            ss.DCOPF.run(solver='CLARABEL')
            ss.DCOPF.export_csv()
            self.assertTrue(os.path.exists(
                os.path.join(tmp, 'pjm5bus_demo_DCOPF.csv')))
            # Should not have leaked to CWD.
            self.assertFalse(os.path.exists('pjm5bus_demo_DCOPF.csv'))
