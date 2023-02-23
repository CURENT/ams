"""
Module for Model class.
"""

import logging
from collections import OrderedDict

from andes.core.modeldata import ModelData as andes_ModelData
from andes.core.model import Model as andes_Model
from andes.core.config import Config

logger = logging.getLogger(__name__)


class Model(andes_ModelData):
    """
    Base class for power system dispatch models.
    """

    def __init__(self, *args, three_params=True, 
                 system=None, config=None,
                 **kwargs) -> None:

        # --- ModelData ---
        super().__init__(*args, three_params=three_params, **kwargs,)

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
