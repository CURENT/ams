"""
Test `Jumper` class.
"""

import unittest

import ams


class TessJumper(unittest.TestCase):
    """
    Test `Jumper` class.
    """

    def setUp(self) -> None:
        """
        Test setup.
        """
        self.sp = ams.load(ams.get_case("5bus/pjm5bus_jumper.xlsx"),
                           setup=True, no_output=True, default_config=True)

    def test_to_andes(self):
        """
        Test `to_andes` method when `Jumper` exists.
        """
        sa = self.sp.to_andes()
        self.assertEqual(sa.Jumper.n, 1)
