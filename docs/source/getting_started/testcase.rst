.. _testcase:

************
Test cases
************

AMS ships with with test cases in the ``ams/cases`` folder.
The cases can be found in the `online repository`_.

.. _`online repository`: https://github.com/CURENT/ams/tree/master/ams/cases

Summary
=======

Below is a summary of the folders and the corresponding test cases. Some folders
contain a README file with notes. When viewing the case folder on GitHub, one
can conveniently read the README file below the file listing.

- ``5bus``: a small PJM 5-bus test case for power flow study [PJM5]_.
- ``ieee14`` and ``ieee39``: the IEEE 14-bus and 39-bus test cases [IEEE]_.
- ``ieee123``: the IEEE 123-bus test case [TSG]_.
- ``matpower``: a subset of test cases from [MATPOWER]_.
- ``npcc`` and ``wecc``: NPCC 140-bus and WECC 179-bus test cases [SciData]_.

How to contribute
=================

We welcome the contribution of test cases! You can make a pull request to
contribute new test cases. Please follow the structure in the ``cases`` folder
and provide an example Jupyter notebook (see ``examples/demonstration``) to
showcase the results of your system.

.. [PJM5] F. Li and R. Bo, "Small test systems for power system economic
        studies," IEEE PES General Meeting, 2010, pp. 1-4, doi:
        10.1109/PES.2010.5589973.
.. [IEEE] University of Washington, "Power Systems Test Case Archive", [Online]. Available:
        https://labs.ece.uw.edu/pstca/
.. [TSG] X. Wang, F. Li, Q. Zhang, Q. Shi and J. Wang, "Profit-Oriented BESS Siting
        and Sizing in Deregulated Distribution Systems," in IEEE Transactions on Smart
        Grid, vol. 14, no. 2, pp. 1528-1540, March 2023, doi: 10.1109/TSG.2022.3150768.
.. [MATPOWER] R. D. Zimmerman, "MATPOWER", [Online]. Available:
        https://matpower.org/
.. [SciData] Q. Zhang and F. Li, “A Dataset for Electricity Market Studies on Western
        and Northeastern Power Grids in the United States,” Scientific Data, vol. 10,
        no. 1, p. 646, Sep. 2023, doi: 10.1038/s41597-023-02448-w.