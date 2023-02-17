from andes.models.group import (GroupBase, ACTopology, StaticGen, ACLine,
                                StaticLoad, StaticShunt, Information, Collection, Undefined)

class Cost(GroupBase):
    def __init__(self):
        super().__init__()
        self.common_params.extend(('gen'))
