"""Common classes amd functions"""
import logging
import pprint

from collections import OrderedDict, defaultdict

logger = logging.getLogger(__name__)


class Config:
    """
    A class to store system, model, and routine configuration
    """

    def __init__(self, name, dct=None, **kwargs) -> None:
        self._name = name
        self._dict = OrderedDict()
        self._help = OrderedDict()
        self._tex = OrderedDict()
        self._alt = OrderedDict()
        self.add(dct, **kwargs)

    def load(self, config):
        """
        Load from a ConfigParser object, ``config``.
        """
        if config is None:
            return
        if self._name in config:
            config_section = config[self._name]
            self.add(OrderedDict(config_section))

    def add(self, dct=None, **kwargs):
        """
        Add config fields from a dictionary or keyword args.

        Existing configs will NOT be overwritten.
        """
        def warn_upper_case(s):
            if any(x.isupper() for x in s):
                logger.warning("Config fields must be in lower case, found %s", s)

        if dct is not None:
            for s in dct.keys():
                warn_upper_case(s)

            self._add(**dct)

        for s in kwargs.keys():
            warn_upper_case(s)

        self._add(**kwargs)

    def add_extra(self, dest, dct=None, **kwargs):
        """
        Add extra contents for config.

        Parameters
        ----------
        dest : str
            Destination string in `_alt`, `_help` or `_tex`.
        dct : OrderedDict, dict
            key: value pairs
        """

        if dct is not None:
            kwargs.update(dct)
        for key, value in kwargs.items():
            if key not in self.__dict__:
                logger.warning("Config field name %s for %s is invalid.", key, dest)
                continue
            self.__dict__[dest][key] = value

    def _add(self, **kwargs):
        """
        Internal function for adding new config keys.

        This function does not perform input data consistency check.
        """

        for key, val in kwargs.items():
            # skip existing entries that are already loaded (from config files)
            if key in self.__dict__:
                continue

            self._set(key, val)

    def _set(self, key, val):
        """
        Set a pair of key and value to the config dict.

        This function does not perform consistency check on input data and will
        not warn of non-existent fields.
        """

        if isinstance(val, str):
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass

        self.__dict__[key] = val

    def as_dict(self, refresh=False):
        """
        Return the config fields and values in an ``OrderedDict``.

        Values are cached in `self._dict` unless refreshed.
        """
        if refresh is True or len(self._dict) == 0:
            out = []
            for key, val in self.__dict__.items():
                if not key.startswith('_'):
                    out.append((key, val))
            self._dict = OrderedDict(out)

        return self._dict

    def __repr__(self):
        return pprint.pformat(self.as_dict())
