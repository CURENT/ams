# Development log

### 2023-05-12

* Draft ``Development`` chapter in documentation
* Collect part of parameters into routiens

NOTE: in a standardized optimization model, the following matrices are used (add the integrality vectors later on):

- c: decision variable coefficients vector
- Aub/Aeq: inequality/equality coefficients matrix
- bub/beq: inequality/equality upper bound vector
- lb/ub: lower/upper bound vector

Next step:

1. define symbolic vars and parameters (in symprocessor)
1. define subtitution map (in symprocessor)
1. formulate and organize matrices: c, Aub, bub, Aeq, beq, lb, ub (in om)
1. solve the problem with optimization solver (in om)
1. retrieve the results to the system (OAlgeb and summary)

### 2023-03-03

* Add ``v`` to ``Algeb``
* Fix ``ipp``
* Mapping results from ``runpf()`` to ``PFlow.run()``

### 2023-03-01

* Add routines to system
* Add ``Algeb`` to routines
* Fix ``ipp``
* Add parser of MATPOWER and PSSE

### 2023-02-25

* Convert system from ``AMS`` to ``ppc``
* [WIP] Refactor PYPOWER

### 2023-02-23

* Fix ``ams.System.setup()``
* Refactor ``ams.model``and models in AMS
* [WIP] Convert system from ``AMS`` to ``ppc``

### 2023-02-22

* Analyze the PYPOWER structure

### 2023-02-21

* Include PYPOWER data file format in documentation

### 2023-2-20

* Fix PYPOWER function

### 2023-02-17

* Import PYPOWER

### 2023-02-16

* Setup base system
* Setup file parser

### 2023-01-26

* Setup versioneer
* Setup documentation

### TODO

- Remove redundant codes in PYPOWER
- Revise docstring of PYPOWER
- Revise docstring of inherited code from ANDES
