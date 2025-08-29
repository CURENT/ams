"""
Test routine export to JSON.
"""
import unittest
import os
import json

import numpy as np

import ams


class TestExportJSON(unittest.TestCase):
    """
    Tests for Routine export to JSON.
    """

    def setUp(self) -> None:
        self.ss = ams.main.load(
            ams.get_case("5bus/pjm5bus_demo.json"),
            default_config=True,
            no_output=True,
        )
        self.expected_json_DCOPF = 'pjm5bus_demo_DCOPF_out.json'
        self.expected_json_ED = 'pjm5bus_demo_ED_out.json'

    def test_no_export(self):
        """
        Test no export when routine is not converged.
        """
        self.assertIsNone(self.ss.DCOPF.export_json())

    def test_export_DCOPF(self):
        """
        Test export DCOPF to JSON.
        """
        self.ss.DCOPF.run(solver='CLARABEL')
        exported_filename = self.ss.DCOPF.export_json()
        self.assertEqual(exported_filename, self.expected_json_DCOPF)
        self.assertTrue(os.path.exists(self.expected_json_DCOPF))

        with open(self.expected_json_DCOPF, 'r') as json_file:
            data = json.load(json_file)

        self.assertIn('StaticGen', data)
        self.assertIn('pg', data['StaticGen'])
        self.assertIn('pmaxe', data['StaticGen'])
        self.assertIn('Line', data)
        self.assertIn('mu1', data['Line'])

    def test_load_DCOPF(self):
        """
        Test loading DCOPF from JSON.
        """
        ss = ams.main.load(
            ams.get_case("5bus/pjm5bus_demo.json"),
            default_config=True,
            no_output=True,
        )
        ss.DCOPF.init()
        ss.DCOPF.load_json(self.expected_json_DCOPF)

        self.assertIsNotNone(ss.DCOPF.pg.v)
        self.assertIsNotNone(ss.DCOPF.plfub.v)
        self.assertIsNotNone(ss.DCOPF.pmaxe.v)
        self.assertIsNotNone(ss.DCOPF.mu1.v)

        os.remove(self.expected_json_DCOPF)

    def test_export_ED(self):
        """
        Test export ED to JSON.
        """
        self.ss.ED.run(solver='CLARABEL')
        self.ss.ED.export_json()
        self.assertTrue(os.path.exists(self.expected_json_ED))

        with open(self.expected_json_ED, 'r') as json_file:
            data = json.load(json_file)

        self.assertIn('StaticGen', data)
        self.assertIn('pg', data['StaticGen'])
        self.assertIn('pmaxe', data['StaticGen'])
        self.assertIn('Line', data)
        self.assertIn('mu1', data['Line'])

    def test_load_ED(self):
        """
        Test loading ED from JSON.
        """
        ss = ams.main.load(
            ams.get_case("5bus/pjm5bus_demo.json"),
            default_config=True,
            no_output=True,
        )
        # NOTE: here we didn't init the ED to test if load_json
        # works without prior initialization
        ss.ED.load_json(self.expected_json_ED)

        self.assertIsNotNone(ss.ED.pg.v)
        np.testing.assert_array_equal(ss.ED.pg.v.shape,
                                      (ss.StaticGen.n, ss.EDTSlot.n))
        self.assertIsNotNone(ss.ED.pmaxe.v)
        np.testing.assert_array_equal(ss.ED.pmaxe.v.shape,
                                      (ss.StaticGen.n, ss.EDTSlot.n))
        self.assertIsNotNone(ss.ED.mu1.v)
        np.testing.assert_array_equal(ss.ED.mu1.v.shape,
                                      (ss.Line.n, ss.EDTSlot.n))

        os.remove(self.expected_json_ED)
