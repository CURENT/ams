"""
Module for the information container models.
"""
from andes.core.param import DataParam
from andes.core.model.modeldata import ModelData

from ams.core.model import Model


class Summary(ModelData, Model):
    """
    Class for storing system summary.
    Can be used for random information or notes.
    """

    def __init__(self, system, config):
        ModelData.__init__(self, three_params=False)

        self.field = DataParam(info='field name')
        self.comment = DataParam(info='information, comment, or anything')
        self.comment2 = DataParam(info='comment field 2')
        self.comment3 = DataParam(info='comment field 3')
        self.comment4 = DataParam(info='comment field 4')

        Model.__init__(self, system, config)
        self.group = 'Information'
