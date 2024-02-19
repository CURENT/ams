.. _input-xlsx:

AMS xlsx
----------

AMS sticks to the same XLSX format as ANDES, where each workbook represents a
model and each row represents a device instance.

Format definition
.................

As indicated in the image below, the AMS xlsx format contains multiple workbooks
(also known as "sheets") shown as tabs at the bottom.
The name of a workbook is a *model* name, and each
workbook contains the parameters of all *devices* that are *instances* of the
model.

.. image:: xlsx.png
   :width: 600
   :alt: Example workbook for Bus

.. note:: Power Flow Data are the bridge between AMS and ANDES cases.
    More discussion can be found in the Examples - Interoperation with ANDES.

For more XLSX format details, please refer to the `ANDES XLSX format <https://docs.andes.app/en/latest/>`_.
