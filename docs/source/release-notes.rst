.. _ReleaseNotes:

=============
Release notes
=============

The APIs before v3.0.0 are in beta and may change without prior notice.

v1.0
==========

v1.0.11 (2025-05-23)
----------------------

- Refactor Documenter
- Fix bug in documentation building

v1.0.10 (2025-05-23)
----------------------

- Add bus type correction in ``system.System.setup()``
- Revise ``ams.io.psse.read`` to complete model Zone when necessary
- Use numerical Area and Zone idx in MATPOWER and PSSE RAW file conversion
- Support JSON format addfile when converting to ANDES case
- Add PSS/E v33 RAW file writer
- In class ``System``, add wrapper methods ``to_mpc``, ``to_m``, ``to_xlsx``,
  ``to_json``, and ``to_raw`` for easier case file export
- Add wrapper routines for PYPOWER: ``DCPF1``, ``PFlow1``, ``DCOPF1``, and ``ACOPF1``
- In routine ``DCOPF`` and its derivatives, add ExpressionCalc ``mu1`` ``mu2`` to
  calculate Lagrange multipliers of line flow limits constraints
- Add wrapper routine ``OPF`` for gurobi_optimods
- Add a demo to show newly added wrapper routines, see
  ``examples/demonstration/demo_wrapper_routines.ipynb``
- Revise ``andes.common.config.Config.update`` to ensure configuration parameters
  are consistently updated in both the object and its internal ``_dict``
- Remove legacy revised PYPOWER module
- Remove function ``ams.shared.ppc2df``

v1.0.9 (2025-04-23)
--------------------

Improve MATPOWER file converter:

- Add a M file case writer
- Include Area and Zone in both MATPOWER read and write

v1.0.8 (2025-04-20)
--------------------

- Run workflow "Publish" only on push tag event
- Include Hawaii synthetic case from
  `Hawaii Synthetic Grid <https://electricgrids.engr.tamu.edu/hawaii40/>`_
- Remove matrices calculation functions in model ``Line``
- Include ``gentype`` and ``genfuel`` when parsing MATPOWER cases
- Fix logging level in ``ACOPF.run``

v1.0.7 (2025-04-14)
--------------------

- Address several wording issues in the documentation
- Switch to ``Area`` from ``Zone`` for zonal calculation
- Extend common parameters in groups ``StaticGen`` and ``StaticLoad`` with ``area``
- Set case ``pjm5bus_demo.xlsx`` as a all-inclusive case
- Include module ``MatProcessor`` in the API documentation
- Improve Line parameters correction in ``System.setup()``
- Make func ``interface._to_andes_pflow`` public
- Discard ``sync_adsys`` step in func ``to_andes_pflow`` to fix mistake in
  parameters conversion
- Update case files

v1.0.6 (2025-04-10)
--------------------

- Enhance handling of Type 1 gencost: Automatically fallback to Type 2 gencost
- Add parameter correction for zero line angle difference

v1.0.5 (2025-04-09)
--------------------

- Include sensitivity matrices calculation demo in documentation
- Add ``DCOPF2``, a PTDF-based DCOPF routine
- Fix bug when update routine parameters before it is initialized

v1.0.4 (2025-04-05)
--------------------

- Fix format in release notes
- Add badges of GitHub relesase and commits in README
- Add a demo to show sensitivity matrices calculation

v1.0.3 (2025-03-17)
--------------------

- Bug fix in function ``ams.interface.parse_addfile``, released in v1.0.3a1

v1.0.2 (2025-02-01)
--------------------

- Enhance the GitHub Actions workflow file
- Deprecate andes logger configuration in ``config_logger``
- Deprecate solver specification in ``demo_ESD1``

v1.0.1 (2025-01-26)
--------------------

Hotfix: removed dependencies on `SCIP` and `pyscipopt` to resolve installation issues

v1.0.0 (2025-01-24)
--------------------

- **Breaking Change**: rename model ``Region`` to ``Zone`` for clarity. Prior case
  files without modification can run into error.
- Fix bugs in ``RTED.dc2ac``
- Minor refacotr ``OptzBase.get_idx`` to reduce duplication
- Rename module ``OptBase`` to ``OptzBase`` for clarity
- Update benchamrk figure in README
- Set ANDES requirement to v1.9.3
- Deprecate method ``get_idx`` and suggest using ``get_all_idxes`` instead
- Remove module ``benchmarks.py`` and its tests for simplicity

Pre-v1.0
==========

v0.9.13 (2024-12-05)
--------------------

- Add a step to report in ``RoutineBase.run``
- Add more tests to cover DG and ES related routines
- Improve formulation for DG and ESD involved routines
- Improve module ``Report`` and method ``RoutineBase.export_csv``
- Support ``TimedEvent`` in ANDES case conversion
- Add Var ``vBus`` in ``DCOPF`` for placeholder

v0.9.12 (2024-11-23)
--------------------

- Refactor ``OModel.initialized`` as a property method
- Add a demo to show using ``Constraint.e`` for debugging
- Fix ``ams.opt.omodel.Param.evaluate`` when its value is a number
- Improve ``ams.opt.omodel.ExpressionCalc`` for better performance
- Refactor module ``ams.opt``
- Add class ``ams.opt.Expression``
- Switch from PYPOWER to ANDES in routine ``PFlow``
- Switch from PYPOWER to regular formulation in routine ``DCPF``
- Refactor routines ``DCPF`` and ``DCOPF``
- In ``RDocumenter``, set Srouce to be owner if there is no src
- Specify ``multiprocess<=0.70.16`` in requirements as 0.70.17 does not support Linux

RC1
~~~~
- Reset setup.py to ensure compatibility

v0.9.11 (2024-11-14)
--------------------

- Add pyproject.toml for PEP 517 and PEP 518 compliance
- Add model ``Jumper``
- Fix deprecation warning related to ``pandas.fillna`` and ``newshape`` in NumPy
- Minor refactor on solvers information in the module ``shared``
- Change default values of minimum ON/OFF duration time of generators to be 1 and 0.5 hours
- Add parameter ``uf`` for enforced generator on/off status
- In servicee ``LoadScale``, consider load online status
- Consider line online status in routine ``ED``
- Add methods ``evaluate`` and ``finalize`` in the class ``OModel`` to handle optimization 
  elements generation and assembling
- Refactor ``OModel.init`` and ``Routine.init``
- Add ANDES paper as the citation file for now
- Add more routine tests for generator trip, line trip, and load trip
- Add a README to overview built-in cases
- Rename methods ``v2`` as ``e`` for classes ``Constraint`` and ``Objective``
- Add benchmark functions
- Improve the usage of ``eval`` in module ``omodel``
- Refactor module ``interop.andes`` as module ``interface`` for simplicity

v0.9.10 (2024-09-03)
--------------------

Hotfix of import issue in ``v0.9.9``.

- In module ``MatProcessor``, add two parameters ``permc_spec`` and ``use_umfpack`` in function ``build_ptdf``
- Follow RTD's deprecation of Sphinx context injection at build time
- In MATPOWER conversion, set devices name as None
- Skip macOS tests in azure-pipelines due to failure in fixing its configuration
- Prepare to support NumPy v2.0.0, but solvers have unexpected behavior
- Improve the logic of setting ``Optz`` value
- Support NumPy v2.0.0

v0.9.9 (2024-09-02)
-------------------

**NOTICE: This version has known issues and has been yanked on PyPI.**

v0.9.8 (2024-06-18)
-------------------

- Assign ``MParam.owner`` when declaring
- In ``MatProcessor``, improve ``build_ptdf`` and ``build_lodf`` to allow partial building and
  incremental building
- Add file ``cases/matpower/Benchmark.json`` for benchmark with MATPOWER
- Improve known good results test
- Minor fix in ``main.py`` selftest part
- Set dependency NumPy version to be <2.0.0 to avoid CVXPY compatibility issues

v0.9.7 (2024-05-24)
-------------------

This patch release add the Roadmap section in the release notes, to list out some potential features.
It also drafts the EV Aggregation model based on the state space modelg, but the finish date remains unknown.

References:

[1] J. Wang et al., "Electric Vehicles Charging Time Constrained Deliverable Provision of Secondary
Frequency Regulation," in IEEE Transactions on Smart Grid, doi: 10.1109/TSG.2024.3356948.

- Fix OTDF calculation
- Add parameter ``dtype='float64'`` and ``no_store=False`` in ``MatProcessor`` PTDF, LODF, and OTDF
  calculation, to save memory
- Add placeholder parameter ``Bus.type``

v0.9.6 (2024-04-21)
-------------------

This patch release refactor and improve ``MatProcessor``, where it support PTDF, LODF,
and OTDF for static analysis.

The reference can be found online "PowerWorld > Web Help > Sensitivities > Line
Outage Distribution Factors".

- Refactor DCPF, PFlow, and ACOPF
- Add a loss factor in ``RTED.dc2ac``
- Add ``DCOPF.dc2ac``
- Fix OModel parse status to ensure no_parsed params can be updated
- Fix and rerun ``ex2``
- Format ``Routine.get`` return type to be consistent with input idx type
- Remove unused ``Routine.prepare``
- Refactor ``MatProcessor`` to separate matrix building
- Add Var ``plf`` in ``DCPF``, ``PFlow``, and ``ACOPF`` to store the line flow
- Add ``build_ptdf``, ``build_lodf``, and ``build_otdf``
- Fix ``Routine.get`` to support pd.Series type idx input
- Reserve ``exec_time`` after ``dc2ac``
- Adjust kloss to fix ``ex2``

v0.9.5 (2024-03-25)
-------------------

- Add more plots in ``demo_AGC``
- Improve line rating adjustment
- Adjust static import sequence in ``models.__init__.py``
- Adjust pjm5bus case line rate_a
- Fix formulation of constraint line angle diff
- Align slack bus angle to zero in ``DCOPF``
- Align StaticGen idx sequence with converted MATPOWER case
- Fix several issues in MATPOWER converter

v0.9.4 (2024-03-16)
-------------------

- Add Var ``pi`` and ExpressionCalc ``pic`` to store the dual of constraint power balance
- Add Param ``M`` and ``D`` to model ``REGCV1``
- Add CPS1 score calculation in ``demo_AGC``

v0.9.3 (2024-03-06)
-------------------

- Major improvemets on ``demo_AGC``
- Bug fix in ``RTED.dc2ac``

v0.9.2 (2024-03-04)
-------------------

- Add ``demo_AGC`` to demonstrate detailed secondary frequency regulation study
- Add ``ExpressionCalc`` to handle post-solving calculation
- Rename ``type='eq'`` to ``is_eq=False`` in ``Constraint`` to avoid overriding built-in attribute
- Several formatting improvements

v0.9.1 (2024-03-02)
-------------------

- Change sphinx extension myst_nb to nbsphinx for math rendering in ``ex8``
- Improve ``symprocessor`` to include routine config
- Add config to Routine reference
- Fix symbol processor issue with power operator

v0.9.0 (2024-02-27)
-------------------

- Add ``ex8`` for formulation customization via API
- Improve Development documentation
- Fix ``addService``, ``addVars``
- Rename ``RoutineModel`` to ``RoutineBase`` for better naming
- Fix ANDES file converter issue
- Initial release on conda-forge

v0.8.5 (2024-01-31)
-------------------

- Improve quality of coverage and format
- Fix dependency issue

v0.8.4 (2024-01-30)
-------------------

- Version cleanup

v0.8.3 (2024-01-30)
-------------------

- Initial release on PyPI

v0.8.2 (2024-01-30)
-------------------

- Improve examples
- Add module ``report`` and func ``RoutineBase.export_csv`` for results export

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
