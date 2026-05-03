import unittest

import os

import ams
from ams.utils.paths import list_cases, get_export_path


class TestPaths(unittest.TestCase):
    def setUp(self) -> None:
        self.npcc = 'npcc/'
        self.matpower = 'matpower/'
        self.ieee14 = ams.get_case("ieee14/ieee14.raw")

    def test_tree(self):
        list_cases(self.npcc, no_print=True)
        list_cases(self.matpower, no_print=True)

    def test_relative_path(self):
        ss = ams.run('ieee14.raw',
                     input_path=ams.get_case('ieee14/', check=False),
                     no_output=True, default_config=True,
                     )
        self.assertNotEqual(ss, None)


class TestExportPath(unittest.TestCase):

    def setUp(self) -> None:
        self.ss = ams.load(ams.get_case('5bus/pjm5bus_demo.json'),
                           setup=True, no_output=True)

    def test_no_fullname(self):
        """
        Test export path with no full name specified.
        """
        self.ss.files.full_name = None

        path, file_name = get_export_path(self.ss,
                                          'DCOPF',
                                          path=None,
                                          fmt='csv')

        dir_path, only_file_name = os.path.split(path)
        self.assertTrue(os.path.exists(dir_path))
        self.assertIsNotNone(only_file_name)
        self.assertEqual(only_file_name, file_name)

    def test_no_path(self):
        """
        Test export path with no path specified.
        """
        path, file_name = get_export_path(self.ss,
                                          'DCOPF',
                                          path=None,
                                          fmt='csv')

        dir_path, only_file_name = os.path.split(path)
        self.assertTrue(os.path.exists(dir_path))
        self.assertIsNotNone(only_file_name)
        self.assertEqual(only_file_name, file_name)

    def test_current_path(self):
        """
        Test export path with current path specified.
        """
        path, file_name = get_export_path(self.ss,
                                          'DCOPF',
                                          path='.',
                                          fmt='csv')

        dir_path, only_file_name = os.path.split(path)
        self.assertTrue(os.path.exists(dir_path))
        self.assertIsNotNone(only_file_name)
        self.assertEqual(only_file_name, file_name)

    def test_path_with_file_name(self):
        """
        Test export path with path and file name specified.
        """
        path, file_name = get_export_path(self.ss,
                                          'DCOPF',
                                          path='./test_export.csv',
                                          fmt='csv',)

        dir_path, only_file_name = os.path.split(path)
        self.assertTrue(os.path.exists(dir_path))
        self.assertEqual(only_file_name, 'test_export.csv')
        self.assertEqual(only_file_name, file_name)
