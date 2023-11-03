import contextlib
import unittest

import ams


class TestRepr(unittest.TestCase):
    """Test __repr__"""
    def setUp(self):
        self.ss = ams.run(ams.get_case("ieee39/ieee39_uced.xlsx"),
                          no_output=True,
                          default_config=True,
                          )

    def test_print_model_repr(self):
        """
        Print out Model ``cache``'s fields and values.
        """
        with contextlib.redirect_stdout(None):
            for model in self.ss.models.values():
                print(model.cache.__dict__)
