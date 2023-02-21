# Development plan

## v1.0

### v1.1

Modeling enhancements

- Multi-timescale dispatch
- Routine of Day Ahead Economic Dispatch (DAED), DAUC
- Coordination of DAED-RTED

### v1.0.0 (2023-05)

Initial release

## Pre-v1.0.0

### v0.9 (2023-04-14)

Function validation and enhancement

- Validation with Matpower
- Matpower input parser

### v0.8 (2023-03-31)

Documentation

- Example of open-loop co-simulation with ANDES
- Example of close-loop co-simulation with ANDES (RTED-TDS)
- Example of fast-prototyping [Virtual Inertia Scheduling - RTED](https://arxiv.org/abs/2209.06677) and co-simulation with ANDES

### v0.7 (2023-03-10)

Routines API design

- DC Optimal Power Flow (DCOPF)
- AC Optimal Power Flow (ACOPF)
- Economic Dispatch (ED)
- Unit Commitment (UC)
- Economic Dispatch (RTED)

TBD: DC-AC conversion for dispatch-dynamic co-simulation

### v0.6.5 (2023-03-03)

Dispatch modeling components

- Dispatch components
- Leveraging ANDES model
- Solver interface to PyPower

### v0.6 (2023-02-24)

Setup solver PYPOWER

### v0.5 (2023-02-17)

Setup base system

Setup development environment

# Development log

Include MATPOWER data file format in documentation (2023-02-21)

Fix PYPOWER functionality (2023-2-20)

Import PYPOWER (2023-02-17)

Setup base system (2023-02-16)

Setup file parser (2023-02-16)

## Development preparation

- Setup versioneer (2023-01-26)
- Setup documentation (2023-01-26)

### TODO

- Remove redundant codes in PYPOWER
- Revise docstring of PYPOWER
- Revise docstring of inherited code from ANDES
