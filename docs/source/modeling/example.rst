Examples
========

One example is provided to demonstrate descriptive dispatch modeling.

DCOPF
----------


DC optimal power flow (DCOPF) is a fundamental routine used in power system analysis.
In this example, we demonstrate how to implement a DCOPF routine in a descriptive
manner using AMS. Below is a simplified DCOPF code snippet.
The full code can be found in :py:mod:`ams.routines.dcopf.DCOPF`.

Data Section
^^^^^^^^^^^^

.. code-block:: python
   :linenos:

      class DCOPF(RoutineBase):

          def __init__(self, system, config):
              RoutineBase.__init__(self, system, config)
              self.info = 'DC Optimal Power Flow'
              self.type = 'DCED'
              # --- Data Section ---
              # --- generator cost ---
              self.c1 = RParam(info='Gen cost coefficient 1',
                               name='c1', tex_name=r'c_{1}', unit=r'$/(p.u.)', 
                               model='GCost', src='c1',
                               indexer='gen', imodel='StaticGen',)
              # --- generator ---
              self.ug = RParam(info='Gen connection status',
                               name='ug', tex_name=r'u_{g}',
                               model='StaticGen', src='u',
                               no_parse=True)
              self.ctrl = RParam(info='Gen controllability',
                                 name='ctrl', tex_name=r'c_{trl}',
                                 model='StaticGen', src='ctrl',
                                 no_parse=True)
              self.ctrle = NumOpDual(info='Effective Gen controllability',
                                     name='ctrle', tex_name=r'c_{trl, e}',
                                     u=self.ctrl, u2=self.ug,
                                     fun=np.multiply, no_parse=True)
              # --- load ---
              self.pd = RParam(info='active demand',
                               name='pd', tex_name=r'p_{d}',
                               model='StaticLoad', src='p0',
                               unit='p.u.',)
              # --- line ---
              self.rate_a = RParam(info='long-term flow limit',
                                   name='rate_a', tex_name=r'R_{ATEA}',
                                   unit='MVA', model='Line',)
              # --- line angle difference ---
              self.amax = NumOp(u=self.rate_a, fun=np.ones_like,
                                rfun=np.dot, rargs=dict(b=np.pi),
                                name='amax', tex_name=r'\theta_{max}',
                                info='max line angle difference',
                                no_parse=True,)
              # --- connection matrix ---
              self.Cg = RParam(info='Gen connection matrix',
                               name='Cg', tex_name=r'C_{g}',
                               model='mats', src='Cg',
                               no_parse=True, sparse=True,)
              # --- system matrix ---
              self.Bbus = RParam(info='Bus admittance matrix',
                                 name='Bbus', tex_name=r'B_{bus}',
                                 model='mats', src='Bbus',
                                 no_parse=True, sparse=True,)

Lines 1-4: Derive subclass ``DCOPF`` from ``RoutineBase``.

Lines 5-6: Define routine information and type.

Lines 9-12: Define linear generator cost coefficients ``c1``, where it
sources from ``GCost.c1`` and sorted by ``StaticGen.gen``.

Lines 22-24: Define effective controllability as service ``ctrle``,
where it multiplies ``ctrl`` and ``ug``.

Lines 42-45: Define generator connection matrix ``Cg``, where it sources from
matrix processor ``mats.Cg`` and is sparse.

Model Section
^^^^^^^^^^^^^

.. code-block:: python
   :linenos:
   :lineno-start: 51

              # --- Model Section ---
              # --- generation ---
              self.pg = Var(info='Gen active power',
                            unit='p.u.',
                            name='pg', tex_name=r'p_g',
                            model='StaticGen', src='p',
                            v0=self.pg0)
              # --- bus ---
              self.aBus = Var(info='Bus voltage angle',
                              unit='rad',
                              name='aBus', tex_name=r'\theta_{bus}',
                              model='Bus', src='a',)
              # --- power balance ---
              pb = 'Bbus@aBus + Pbusinj + Cl@pd + Csh@gsh - Cg@pg'
              self.pb = Constraint(name='pb', info='power balance',
                                   e_str=pb, type='eq',)
              # --- line flow ---
              self.plf = Var(info='Line flow',
                             unit='p.u.',
                             name='plf', tex_name=r'p_{lf}',
                             model='Line',)
              self.plflb = Constraint(info='line flow lower bound',
                                      name='plflb', type='uq',
                                      e_str='-Bf@aBus - Pfinj - rate_a',)
              self.plfub = Constraint(info='line flow upper bound',
                                      name='plfub', type='uq',
                                      e_str='Bf@aBus + Pfinj - rate_a',)
              # --- objective ---
              obj = 'sum(mul(c2, power(pg, 2)))'
              obj += '+ sum(mul(c1, pg))'
              obj += '+ sum(mul(ug, c0))'
              self.obj = Objective(name='obj',
                                   info='total cost', unit='$',
                                   sense='min', e_str=obj,)

Continued from the above code.

Lines 53-57: Define variable ``pg``, where it links to ``StaticGen.p``
and initial value ``pg0``.

Lines 68-71: Define variable ``plf``, where it links to ``Line`` with
no target source variable nor initial value.

Lines 72-77: Define inequality constraints ``plflb`` and ``plfub``
for line flow limits.

Lines 79-84: Define objective function ``obj`` for minimizing total cost.

Finalize
^^^^^^^^

Lastly, similar to finalize a device model, we need to finalize the routine
by adding the ``RTED`` to the routine list in ``$HOME/ams/ams/routines/__init__.py``,
where 'rted' is the file name, and 'RTED' is the routine name.

.. code-block:: python

      all_routines = OrderedDict([
            ... ...
            ('dcopf', ['DCOPF']),
            ('ed', ['ED', 'EDDG', 'EDES']),
            ('rted', ['RTED', 'RTEDDG', 'RTEDES', 'RTEDVIS']),
            ... ...
      ])

.. note::
      Refer to the documentation "Example - Customize Formulation"
      for API customization that does not require modification of the source code.