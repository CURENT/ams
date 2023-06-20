Interoperation
======================

This section introduces the interoperation between AMS and dynamic simulation
engine ANDES.

File conversion
--------------------

.. autofunction:: ams.interop.andes.to_andes
    :noindex:

Data mapping
-----------------


Data mapping in the ams.interop.andes.
Dynamic module is responsible for the conversion and synchronization of data
between AMS and ANDES.
It utilizes two methods: ``send`` and ``receive``.

When using this interface, it automatically determines whether to use the
dynamic or static model based on the initialization status of the TDS.
You can refer to the source code of :py:mod:`ams.interop.andes.Dynamic.send`
and :py:mod:`ams.interop.andes.Dynamic.receive`
for more in-depth information on their implementation.

Check ANDES documentation
`StaticGen <https://docs.andes.app/en/latest/groupdoc/StaticGen.html#staticgen>`_
for more details about substituting static generators with dynamic generators.

.. autoclass:: ams.interop.andes.Dynamic
    :noindex:
    :members: send, receive