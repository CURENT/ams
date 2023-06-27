Model
=========

This section introduces the modeling of power system devices. Here the term
``model`` refers to the descriptive model of a device, which is used to
hold the model-level data and variables, such as ``Bus``, ``Line``, and ``PQ``.

AMS follows the model organization design of ANDES, where two classes
defined in ANDES, ``ModelData`` and ``Model``, are used.

Parameters
------------
Parameter is an atom element in building a power system model. Most parameters
are read from an input file, and other parameters are calculated from the existing
parameters.

AMS leverages the parameter definition in ANDES, where four classes,
``DataParam``, ``IdxParam``, ``NumParam``, and ``ExtParam`` are used.
More details can be found in ANDES documentation
`Development - Parameters <https://docs.andes.app/en/latest/modeling/parameters.html>`_.

Variables
-----------

In AMS, the definition of variables ``Algeb`` is simplified from ANDES. The
``Algeb`` class is used to define algebraic variables in the model level, which are used to
exchange data with dynamic simulator.

.. autoclass:: ams.core.var.Algeb
    :noindex:

      Algeb

.. note::

    The ``Algeb`` class here is not directly used for optimization purpose, we will discuss
    its role further in the Routine section.

Model
--------------

In AMS, a "Model" contains two parts, ``ModelData`` and ``Model``. The ``ModelData`` holds
the model-level parameters, and the ``Model`` holds the model-level variables.

``Model`` is simplified from ANDES, where only Algebs-related definitions are kept. As for
the ``ModelData``, most of the existing definitions in ANDES are ready to use, with some
minor modifications.

.. autoclass:: ams.core.model
    :noindex:

      Model


Examples
------------

The following two examples demonstrate how to define a device model in AMS.


PV model
^^^^^^^^^^^^^^

In this example, we define a ``PV`` model in three steps, data definition, model definition,
and manufacturing.

First, we need to define the parameters not included in ANDES ``PVData``. In this example,
we hold the parameters in a separate class ``GenParam``.

.. code-block:: python

    from andes.core.param import NumParam, ExtParam

    class GenParam:
        def __init__(self) -> None:
            self.Pc1 = NumParam(default=0.0,
                                info="lower real power output of PQ capability curve",
                                tex_name=r'P_{c1}',
                                unit='p.u.')
            self.Pc2 = NumParam(default=0.0,
                                info="upper real power output of PQ capability curve",
                                tex_name=r'P_{c2}',
                                unit='p.u.')

            ......

            self.pg0 = NumParam(default=0.0,
                                info='real power start point',
                                tex_name=r'p_{g0}',
                                unit='p.u.',
                                )

Second, we define the ``PVModel`` model with two algebraic variables and a external parameter.

.. code-block:: python

    from ams.core.model import Model
    from ams.core.var import Algeb  # NOQA

    class PVModel(Model):

        def __init__(self, system=None, config=None):
            super().__init__(system, config)
            self.group = 'StaticGen'

            self.zone = ExtParam(model='Bus', src='zone', indexer=self.bus, export=False,
                                info='Retrieved zone idx', vtype=str, default=None,
                                )
            self.p = Algeb(info='actual active power generation',
                          unit='p.u.',
                          tex_name='p',
                          name='p',
                          )
            self.q = Algeb(info='actual reactive power generation',
                          unit='p.u.',
                          tex_name='q',
                          name='q',
                          )

.. note::

    The external parameter ``zone`` is added here to enable the zonal reserve dispatch, and it
    is not included in ANDES ``PV``.

Third, we manufacture these classes together as the ``PV`` model.

.. code-block:: python

    from andes.models.static.pv import PVData  # NOQA

    class PV(PVData, GenParam, PVModel):
        """
        PV generator model.

        TODO: implement type conversion in config
        """

        def __init__(self, system, config):
            PVData.__init__(self)
            GenParam.__init__(self)
            PVModel.__init__(self, system, config)

Lastly, we need to finalize the model by adding the ``PV`` model to the model list in
``$HOME/ams/ams/models/__init__.py``, where 'static' is the file name, and 'PV' is the
model name.

.. code-block:: python

    ams_file_classes = list([
        ('info', ['Summary']),
        ('bus', ['Bus']),
        ('static', ['PQ', 'PV', 'Slack']),
        ... ...
    ])

.. note::

    The model development procedures is similar to ANDES. The only difference is that
    a dispatch model is much simpler than a dynamic model. In the dispatch model, we only
    defines the data and small amount of variables.
    Further details for dynamic model development can be found in ANDES documentation
    `Development - Examples <https://docs.andes.app/en/latest/modeling/examples.html>`_.

Line model
^^^^^^^^^^^^^^

TODO.
In this example, we define a ``Line`` model, where the data is extended from existing
ANDES ``LineData`` by including two parameters ``amin`` and ``amax``.
