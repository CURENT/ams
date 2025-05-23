import logging

from andes.core.common import Config as AndesConfig

logger = logging.getLogger(__name__)


class Config(AndesConfig):
    """
    A class for storing configuration, can be used in system,
    model, routine, and other modules.

    Revised from `andes.core.common.Config`.
    """

    def __init__(self, name, dct=None, **kwargs):
        super().__init__(name, dct, **kwargs)

    def update(self, dct: dict = None, **kwargs):
        """
        Update the configuration from a file.
        """
        if dct is not None:
            kwargs.update(dct)

        for key, val in kwargs.items():
            self._set(key, val)
            self._dict[key] = val

        self.check()
