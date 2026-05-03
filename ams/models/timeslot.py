"""
Models for multi-period scheduling.

The TimeSlot family is split into two layers:

1. **Slot definitions** — :class:`EDSlot` and :class:`UCSlot` carry
   one row per time slot (``idx``, ``name``, ``u``). They define the
   horizon used by routine Vars (e.g. ``self.pg.horizon =
   self.timeslot``).

2. **Per-slot data tables** — :class:`EDSlotLoad`, :class:`EDSlotGen`,
   :class:`UCSlotLoad` carry one row per ``(device, slot)`` pair with
   a scalar value column. Routines pivot these into 2D matrices via
   :class:`ams.core.param.RParam` with ``horizon=`` (mirroring the
   :attr:`ams.opt.var.Var.horizon` pattern used on the output side).

This shape replaces the pre-v1.3.0 single-table design where
``EDSlot`` carried CSV-list cells whose positional ordering
implicitly aligned with ``StaticGen.get_all_idxes()`` /
``Area.get_all_idxes()``. The implicit contract is now explicit
through ``IdxParam`` columns.
"""

from andes.core import ModelData, NumParam, IdxParam

from ams.core.model import Model


class EDSlot(ModelData, Model):
    """
    Time slot definitions for multi-period ED.

    One row per slot. Per-slot data lives on :class:`EDSlotLoad` (load
    scaling) and :class:`EDSlotGen` (generator commitment).
    """

    def __init__(self, system=None, config=None):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Horizon'


class UCSlot(ModelData, Model):
    """
    Time slot definitions for multi-period UC.

    One row per slot. Per-slot load-scaling data lives on
    :class:`UCSlotLoad`. UC solves for generator commitment, so no
    per-slot ``ug`` table is needed.
    """

    def __init__(self, system=None, config=None):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Horizon'


class EDSlotLoad(ModelData, Model):
    """
    Per-(area, slot) area load scaling factor for ED.

    .. versionadded:: 1.3.0
       Replaces the CSV-list ``EDSlot.sd`` cell. Each row carries one
       scalar ``sd`` keyed by ``(area, slot)``.
    """

    def __init__(self, system=None, config=None):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Horizon'

        self.area = IdxParam(model='Area',
                             info='area idx (primary key)',
                             mandatory=True,
                             )
        self.slot = IdxParam(model='EDSlot',
                             info='time slot idx (secondary key)',
                             mandatory=True,
                             )
        self.sd = NumParam(default=1.0,
                           info='area load scaling factor',
                           tex_name=r's_{d}',
                           vtype=float,
                           )


class EDSlotGen(ModelData, Model):
    """
    Per-(gen, slot) generator commitment decision for ED.

    .. versionadded:: 1.3.0
       Replaces the CSV-list ``EDSlot.ug`` cell. Each row carries one
       scalar ``ug`` keyed by ``(gen, slot)``.
    """

    def __init__(self, system=None, config=None):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Horizon'

        self.gen = IdxParam(model='StaticGen',
                            info='generator idx (primary key)',
                            mandatory=True,
                            )
        self.slot = IdxParam(model='EDSlot',
                             info='time slot idx (secondary key)',
                             mandatory=True,
                             )
        self.ug = NumParam(default=1,
                           info='unit commitment decision',
                           tex_name=r'u_{g}',
                           vtype=int,
                           )


class UCSlotLoad(ModelData, Model):
    """
    Per-(area, slot) area load scaling factor for UC.

    .. versionadded:: 1.3.0
       Replaces the CSV-list ``UCSlot.sd`` cell. Each row carries one
       scalar ``sd`` keyed by ``(area, slot)``.
    """

    def __init__(self, system=None, config=None):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Horizon'

        self.area = IdxParam(model='Area',
                             info='area idx (primary key)',
                             mandatory=True,
                             )
        self.slot = IdxParam(model='UCSlot',
                             info='time slot idx (secondary key)',
                             mandatory=True,
                             )
        self.sd = NumParam(default=1.0,
                           info='area load scaling factor',
                           tex_name=r's_{d}',
                           vtype=float,
                           )
