import logging

from andes.core.common import Config as adConfig

logger = logging.getLogger(__name__)


class Config(adConfig):
    """
    A class for storing configuration, can be used in system,
    model, routine, and other modules.

    Revised from `andes.core.common.Config`, where update method
    is modified to ensure values in the dictionary are set
    in the configuration object.

    .. versionadded:: 1.0.11
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
