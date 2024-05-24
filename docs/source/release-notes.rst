.. _ReleaseNotes:

=============
Release notes
=============

The APIs before v3.0.0 are in beta and may change without prior notice.

Pre-v1.0.0
==========

v0.9.7 (2024-05-24)
-------------------

This patch release add the Roadmap section in the release notes, to list out some potential features.
It also drafts the EV Aggregation model based on the state space modelg, but the finish date remains unknown.

References:

[1] J. Wang et al., "Electric Vehicles Charging Time Constrained Deliverable Provision of Secondary
Frequency Regulation," in IEEE Transactions on Smart Grid, doi: 10.1109/TSG.2024.3356948.

- Fix OTDF calculation
- Add parameter `dtype='float64'` and `no_store=False` in `MatProcessor` PTDF, LODF, and OTDF calculation, to save memory
- Add placeholder parameter `Bus.type`

v0.9.6 (2024-04-21)
-------------------

This patch release refactor and improve `MatProcessor`, where it support PTDF, LODF,
and OTDF for static analysis.

The reference can be found online "PowerWorld > Web Help > Sensitivities > Line
Outage Distribution Factors".

- Refactor DCPF, PFlow, and ACOPF
- Add a loss factor in ``RTED.dc2ac()``
- Add ``DCOPF.dc2ac()``
- Fix OModel parse status to ensure no_parsed params can be updated
- Fix and rerun ex2
- Format ``Routine.get()`` return type to be consistent with input idx type
- Remove unused ``Routine.prepare()``
- Refactor `MatProcessor` to separate matrix building
- Add Var `plf` in `DCPF`, `PFlow`, and `ACOPF` to store the line flow
- Add `build_ptdf`, `build_lodf`, and `build_otdf`
- Fix ``Routine.get()`` to support pd.Series type idx input
- Reserve `exec_time` after ``dc2ac()``
- Adjust kloss to fix ex2

v0.9.5 (2024-03-25)
-------------------

- Add more plots in demo_AGC
- Improve line rating adjustment
- Adjust static import sequence in `models.__init__.py`
- Adjust pjm5bus case line rate_a
- Fix formulation of constraint line angle diff
- Align slack bus angle to zero in `DCOPF`
- Align StaticGen idx sequence with converted MATPOWER case
- Fix several issues in MATPOWER converter

v0.9.4 (2024-03-16)
-------------------

- Add Var ``pi`` and ExpressionCalc ``pic`` to store the dual of constraint power balance
- Add Param ``M`` and ``D`` to model ``REGCV1``
- Add CPS1 score calculation in demo_AGC

v0.9.3 (2024-03-06)
-------------------

- Major improvemets on demo_AGC
- Bug fix in ``RTED.dc2ac()``

v0.9.2 (2024-03-04)
-------------------

- Add demo_AGC to demonstrate detailed SFR study
- Add ``ExpressionCalc`` to handle post-solving calculation
- Rename ``type='eq'`` to ``is_eq=False`` in ``Constraint`` to avoid overriding built-in attribute
- Several formatting improvements

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

Roadmap
=======

This section lists out some potential features that may be added in the future.
Note that the proposed features are not guaranteed to be implemented and subject to change.

Electric Vehicle for Grid Service
------------------------------------------

A charging-time-constrained EV aggregation based on the state-space model

References:

[1] J. Wang et al., "Electric Vehicles Charging Time Constrained Deliverable Provision of Secondary
Frequency Regulation," in IEEE Transactions on Smart Grid, doi: 10.1109/TSG.2024.3356948.

[2] M. Wang et al., "State Space Model of Aggregated Electric Vehicles for Frequency Regulation," in
IEEE Transactions on Smart Grid, vol. 11, no. 2, pp. 981-994, March 2020, doi: 10.1109/TSG.2019.2929052.

Distribution OPF
--------------------------

- Distribution networks OPF and its LMP
- DG siting and sizing considering energy equity

References:

[1] H. Yuan, F. Li, Y. Wei and J. Zhu, "Novel Linearized Power Flow and Linearized OPF Models for
Active Distribution Networks With Application in Distribution LMP," in IEEE Transactions on Smart Grid,
vol. 9, no. 1, pp. 438-448, Jan. 2018, doi: 10.1109/TSG.2016.2594814.

[2] C. Li, F. Li, S. Jiang, X. Wang and J. Wang, "Siting and Sizing of DG Units Considering Energy
Equity: Model, Solution, and Guidelines," in IEEE Transactions on Smart Grid, doi: 10.1109/TSG.2024.3350914.

Planning
--------------------------

- Transmission expansion planning
