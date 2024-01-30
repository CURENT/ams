import unittest

import ams
from ams.utils.paths import list_cases


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
