"""
Module for Model class.
"""

import logging
from collections import OrderedDict

from andes.core.model.model import Model as andes_Model
from andes.core.common import Config

logger = logging.getLogger(__name__)


class Model(andes_Model):
    """
    Base class for power system dispatch models.
    """

    def __init__(self, system=None, config=None):

        # --- Model ---
        self.system = system
        self.group = 'Undefined'
        self.config = Config(name=self.class_name)  # `config` that can be exported
        if config is not None:
            self.config.load(config)

        # basic configs
        self.config.add(OrderedDict((('allow_adjust', 1),
                                    ('adjust_lower', 0),
                                    ('adjust_upper', 1),
                                     )))

        # TODO: duplicate from ANDES, disable for now
        # self.syms = SymProcessor(self)  # symbolic processor instance
        # self.docum = Documenter(self)
