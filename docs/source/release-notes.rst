.. _ReleaseNotes:

=============
Release notes
=============

The APIs before v3.0.0 are in beta and may change without prior notice.

Pre-v1.0.0
==========
v0.6.2
-------------------

- Enhance docstring
- Remove unused module ``utils.LazyImport``

v0.6.1 (2023-03-05)
-------------------

- Fix incompatiability of NumPy attribute ``object`` in  ``io.matpower._get_bus_id_caller``
- Add file parser ``io.pypower`` for PYPOWER case file
- Deprecate PYPOWER interface ``solvers.ipp``

v0.6 (2023-03-04)
-------------------

- Set up PYPOWER for power flow calculation
- Add PYPOWER interface ``solvers.ipp``
- Develop module ``routines`` for routine analysis
- Revise module ``system``, ``core.var``, ``core.model`` for routine analysis
- Set up routine ``PFlow`` for power flow calculation
- Add file parser ``io.matpower`` and ``io.raw`` for MATPOWER file and RAW file
- Documentation of APIs

v0.5 (2023-02-17)
-------------------

- Base System setup
- Development preparation

v0.4 (2023-01)
-------------------

This release outlines the package.