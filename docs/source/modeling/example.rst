Examples
========

Two examples are provided to demonstrate the usage of the interopreable dispatch
modeling.

DCOPF
-----------

DC optimal power flow (DCOPF) is one of the basic routine used in power system analysis. In this example,
we will show how to implement a DCOPF routine.

Defines routine data
^^^^^^^^^^^^^^^^^^^^

``DCOPFData`` used in a routine.

.. code-block:: python

      class DCOPFData(RoutineData):
      """
      DCOPF data.
      """

      def __init__(self):
            RoutineData.__init__(self)
            # --- generator cost ---
            self.c2 = RParam(info='Gen cost coefficient 2',
                              name='c2',
                              tex_name=r'c_{2}',
                              unit=r'$/(p.u.^2)',
                              owner_name='GCost',
                              )
            self.c1 = RParam(info='Gen cost coefficient 1',
                              name='c1',
                              tex_name=r'c_{1}',
                              unit=r'$/(p.u.)',
                              owner_name='GCost',
                              )
            self.c0 = RParam(info='Gen cost coefficient 0',
                              name='c0',
                              tex_name=r'c_{0}',
                              unit=r'$',
                              owner_name='GCost',
                              )
            # --- generator limit ---
            self.pmax = RParam(info='generator maximum active power in system base',
                              name='pmax',
                              tex_name=r'p_{max}',
                              unit='p.u.',
                              owner_name='StaticGen',
                              )
            self.pmin = RParam(info='generator minimum active power in system base',
                              name='pmin',
                              tex_name=r'p_{min}',
                              unit='p.u.',
                              owner_name='StaticGen',
                              )
            # --- load ---
            # NOTE: following two parameters are temporary solution
            self.pd1 = RParam(info='active power load in system base in gen bus',
                              name='pd1',
                              tex_name=r'p_{d1}',
                              unit='p.u.',
                              )
            self.pd2 = RParam(info='active power load in system base in non-gen bus',
                              name='pd2',
                              tex_name=r'p_{d2}',
                              unit='p.u.',
                              )
            # --- line ---
            self.rate_a = RParam(info='long-term flow limit flow limit',
                              name='rate_a',
                              tex_name=r'R_{ATEA}',
                              unit='MVA',
                              owner_name='Line',
                              )
            self.PTDF1 = RParam(info='PTDF matrix 1',
                              name='PTDF1',
                              tex_name=r'P_{TDF1}',
                              )
            self.PTDF2 = RParam(info='PTDF matrix 2',
                              name='PTDF2',
                              tex_name=r'P_{TDF2}',
                              )

Defines routine model
^^^^^^^^^^^^^^^^^^^^^^

``DCOPFModel`` used in a routine.

.. code-block:: python

      class DCOPFModel(DCOPFBase):
      """
      DCOPF model.
      """

      def __init__(self, system, config):
            DCOPFBase.__init__(self, system, config)
            self.info = 'DC Optimal Power Flow'
            self.type = 'DCED'
            # --- vars ---
            self.pg = Var(info='actual active power generation',
                        unit='p.u.',
                        name='pg',
                        src='p',
                        tex_name=r'p_{g}',
                        owner_name='StaticGen',
                        lb=self.pmin,
                        ub=self.pmax,
                        )
            # --- constraints ---
            self.pb = Constraint(name='pb',
                              info='power balance',
                              e_str='sum(pd1) + sum(pd2) - sum(pg)',
                              type='eq',
                              )
            self.lub = Constraint(name='lub',
                                    info='line limits upper bound',
                                    e_str='PTDF1 @ (pg - pd1) - PTDF2 * pd2 - rate_a',
                                    type='uq',
                                    )
            self.llb = Constraint(name='llb',
                                    info='line limits lower bound',
                                    e_str='- PTDF1 @ (pg - pd1) + PTDF2 * pd2 - rate_a',
                                    type='uq',
                                    )
            # --- objective ---
            self.obj = Objective(name='tc',
                              info='total generation cost',
                              e_str='sum(c2 * pg**2 + c1 * pg + c0)',
                              sense='min',)

Manufacture routine model
^^^^^^^^^^^^^^^^^^^^^^^^^^

``DCOPF`` is the manufactured DCOPF routine.

.. code-block:: python

      class DCOPF(DCOPFData, DCOPFModel):
      """
      Standard DC optimal power flow (DCOPF).
      """

      def __init__(self, system, config):
            DCOPFData.__init__(self)
            DCOPFModel.__init__(self, system, config)

Finalize
^^^^^^^^

``finalize`` is used to finalize the routine.

RTED
-----------

TODO. Real-time economic dispatch (RTED) is the base routine used to interface with
the dynamic simulator. In this example, we will show how to extend the existing DCOPF
routine to the desired RTED routine.