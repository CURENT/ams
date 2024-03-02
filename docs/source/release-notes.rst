.. _ReleaseNotes:

=============
Release notes
=============

The APIs before v3.0.0 are in beta and may change without prior notice.

Pre-v1.0.0
==========

v0.9.1 (2024-03-02)
-------------------

- Change sphinx extension myst_nb to nbsphinx for math rendering in ex8
- Improve ``symprocessor`` to include routine config
- Add config to Routine reference
- Fix symbol processor issue with power operator

v0.9.0 (2024-02-27)
-------------------

- Add ex8 to demonstrate customize existing formulations via API
- Improve Development documentation
- Fix ``addService``, ``addVars``
- Rename ``RoutineModel`` to ``RoutineBase`` for better naming
- Fix ANDES file converter issue
- Initial release to conda-forge

v0.8.5 (2024-01-31)
-------------------

- Improve quality of coverage and format
- Fix dependency issue

v0.8.4 (2024-01-30)
-------------------

- Version cleanup

v0.8.3 (2024-01-30)
-------------------

- Initial release to PyPI

v0.8.2 (2024-01-30)
-------------------

- Improve examples
- Add report module and export_csv for results export

v0.8.1 (2024-01-20)
-------------------

- Improve ``MatProcessor``
- Add more examples
- Improve ANDES interface

v0.8.0 (2024-01-09)
-------------------

- Refactor ``DCED`` routines to improve performance

v0.7.5 (2023-12-28)
-------------------

- Refactor ``MatProcessor`` and ``DCED`` routines to improve performance
- Integrate sparsity pattern in ``RParam``
- Rename energy storage routines ``RTED2``, ``ED2`` and ``UC2`` to ``RTEDES``, ``EDES`` and ``UCES``

v0.7.4 (2023-11-29)
-------------------

- Refactor routins and optimization models to improve performance
- Fix routines modeling
- Add examples
- Fix built-in cases

v0.7.3 (2023-11-03)
-------------------

- Add tests

v0.7.2 (2023-10-26)
-------------------

- Add routines ``ED2`` and ``UC2``
- Minor fix on ``SymProcessor`` and ``Documenter``

v0.7.1 (2023-10-12)
-------------------

- Add function ``_initial_guess`` to routine ``UC``
- Refactor PYPOWER

v0.7.0 (2023-09-22)
-------------------

- Add interfaces for customizing optimization
- Add models ``REGCV1`` and ``REGCV1Cost`` for virtual inertia scheduling
- Add cost models: ``SRCost``, ``NSRCost``, ``DCost``
- Add reserve models: ``SR``, ``NSR``
- Add routine ``UC``
- Add routine ``RTED2`` to include energy storage model

v0.6.7 (2023-08-02)
-------------------

- Version cleanup

v0.6.6 (2023-07-27)
-------------------

- Improve routine reference
- Add routine ED, LDOPF

v0.6.5 (2023-06-27)
-------------------

- Update documentation with auto-generated model and routine reference
- Add interface with ANDES ``ams.interop.andes``
- Add routine RTED and example of RTED-TDS co-simulation
- Draft development documentation

v0.6.4 (2023-05-23)
-------------------

- Setup PFlow and DCPF using PYPOWER

v0.6.3 (2023-05-22)
-------------------

- Using CVXPY for draft implementation
- Improve ``model``, ``group``, ``param`` and ``var`` in ``core``
- Refactor ``routines`` and ``opt``
- Improve PYPOWER interface ``io.pypower.system2ppc``
- Fix PYPOWER function ``solver.pypower.makePTDF``

v0.6.2 (2023-04-23)
-------------------

- Enhance docstring
- Remove unused module ``utils.LazyImport``
- Remove unused module ``shared``

v0.6.1 (2023-03-05)
-------------------

- Fix incompatiability of NumPy attribute ``object`` in  ``io.matpower._get_bus_id_caller``
- Add file parser ``io.pypower`` for PYPOWER case file
- Deprecate PYPOWER interface ``solvers.ipp``

v0.6.0 (2023-03-04)
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

- Develop module ``system``, ``main``, ``cli``
- Development preparation: versioneer, documentation, etc.

v0.4 (2023-01)
-------------------

This release outlines the package.