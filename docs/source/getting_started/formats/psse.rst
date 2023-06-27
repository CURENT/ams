
.. _psse:

PSS/E RAW
-----------------

The Siemens PSS/E data format is a widely used for power system simulation.
PSS/E uses a variety of plain-text files to store data for different actions.
The RAW format (with file extension ``.raw``) is used to store the steady-state
data for power flow analysis.
Leveraging ANDES PSS/E parser, one can load PSS/E RAW files into AMS for power
flow study.

RAW Compatibility
.................
AMS supports PSS/E RAW in versions 32 and 33. Newer versions of
``raw`` files can store PSS/E settings along with the system data, but such
feature is not yet supported in AMS. Also, manually edited ``raw`` files can
confuse the parser in AMS. Following manual edits, it is strongly recommended
to load the data into PSS/E and save the case as a v33 RAW file.

AMS supports most power flow models in PSS/E. It needs to be recognized that
the power flow models in PSS/E is is a larger set compared with those in AMS.
For example, switched shunts in PSS/E are converted to fixed ones, not all
three-winding transformer flags are supported, and HVDC devices are not yet
converted. This is not an exhaustive list, but all of them are advanced models.

We welcome contributions but please also reach out to us if you need
to arrange the development of such models.

Loading files
.............

In the command line, PSS/E files can be loaded with

.. code-block:: bash

    ams run kundur.raw

Likewise, one can convert PSS/E files to AMS xlsx:

.. code-block:: bash

    ams run kundur.raw -c

This will convert all models in the RAW files. 

To load PSS/E files into a scripting environment, see Example - "Working with
Data".
