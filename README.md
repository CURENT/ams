# LTB AMS

Python Software for Power System Scheduling Modeling and Co-Simulation with Dynamics, serving as the market simulator for the [CURENT Largescale Testbed][LTB Repository].

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://github.com/CURENT/ams/blob/master/LICENSE)
![platforms](https://anaconda.org/conda-forge/ltbams/badges/platforms.svg)
[![Python Versions](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![DOI:10.1109/TSTE.2025.3528027](https://zenodo.org/badge/DOI/10.1109/TSTE.2025.3528027.svg)](https://ieeexplore.ieee.org/document/10836855)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![codecov](https://codecov.io/gh/CURENT/ams/graph/badge.svg?token=RZI5GLLBQH)](https://codecov.io/gh/CURENT/ams)
![Repo Size](https://img.shields.io/github/repo-size/CURENT/ams)
[![GitHub last commit (master)](https://img.shields.io/github/last-commit/CURENT/ams/master?label=last%20commit%20to%20master)](https://github.com/CURENT/ams/commits/master/)
[![GitHub last commit (develop)](https://img.shields.io/github/last-commit/CURENT/ams/develop?label=last%20commit%20to%20develop)](https://github.com/CURENT/ams/commits/develop/)
[![libraries](https://img.shields.io/librariesio/release/pypi/ltbams)](https://libraries.io/pypi/ltbams)
[![Structure](https://img.shields.io/badge/code_base-visualize-blue)](https://mango-dune-07a8b7110.1.azurestaticapps.net/?repo=CURENT%2Fams)

[![Compatibility Tests](https://github.com/CURENT/ams/actions/workflows/compatibility.yml/badge.svg)](https://github.com/CURENT/ams/actions/workflows/compatibility.yml)
[![Publish to TestPyPI and PyPI](https://github.com/CURENT/ams/actions/workflows/publish-pypi.yml/badge.svg?branch=master)](https://github.com/CURENT/ams/actions/workflows/publish-pypi.yml)
[![Azure Pipline](https://dev.azure.com/curentltb/ams/_apis/build/status%2FCURENT.ams?branchName=master)](https://dev.azure.com/curentltb/ams/_build/latest?definitionId=2&branchName=master)

<img src="docs/source/images/sponsors/CURENT_Logo_NameOnTrans.png" alt="CURENT ERC Logo" width="300" height="auto">

|               | Stable                                                                                                                                        | Latest                                                                                                                                        |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Documentation | [![Documentation Status](https://readthedocs.org/projects/ams/badge/?version=stable)](https://ams.readthedocs.io/en/stable/?badge=stable) | [![Latest Documentation](https://readthedocs.org/projects/ams/badge/?version=latest)](https://ams.readthedocs.io/en/latest/?badge=latest) | 


| Badges        |                                                                                                                                                                                                                                                     |                                                                                                                                                                                                            |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Downloads     | [![PyPI Version](https://img.shields.io/pypi/v/ltbams.svg)](https://pypi.python.org/pypi/ltbams)         | [![Conda Version](https://anaconda.org/conda-forge/ltbams/badges/version.svg)](https://anaconda.org/conda-forge/ltbams) |
| Try on Binder | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/curent/ams/master)                                                                                                                                                 |                                                                                                                                                                                                            |
| Code Quality  |[![Codacy Badge](https://app.codacy.com/project/badge/Grade/69456da1b8634f2f984bd769e35f0050)](https://app.codacy.com/gh/CURENT/ams/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)| [![Codacy Badge](https://app.codacy.com/project/badge/Coverage/69456da1b8634f2f984bd769e35f0050)](https://app.codacy.com/gh/CURENT/ams/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage) |



# Why AMS

AMS facilitates **Dynamics Incorporated Scheduling** and **Scheduling-Dynamics Co-Simulation**
through an integrated interface with ANDES.

This package is helpful for power system engineers, researchers, and students conducting
scheduling and transient stability studies at specific operating points. It also benefits
those interested in developing new scheduling formulations and algorithms, particularly
by extending existing formulations to include new decision variables, constraints, and
objective functions.

AMS is a **Modeling Framework** that provides a descriptive way to formulate
scheduling problems. The optimization problems are then handled by **CVXPY**
and solved with third-party solvers.

AMS produces credible scheduling results and competitive performance.
The following results show the comparison of DCOPF between AMS and other tools.

| Cost [\$]             | AMS          | pandapower   | MATPOWER     |
|-----------------------|--------------|--------------|--------------|
| IEEE 14-Bus           | 7,642.59     | 7,642.59     | 7,642.59     |
| IEEE 39-Bus           | 41,263.94    | 41,263.94    | 41,263.94    |
| PEGASE 89-Bus         | 5,733.37     | 5,733.37     | 5,733.37     |
| IEEE 118-Bus          | 125,947.88   | 125,947.88   | 125,947.88   |
| NPCC 140-Bus          | 810,033.37   | 810,016.06   | 810,033.37   |
| WECC 179-Bus          | 411,706.13   | 411,706.13   | 411,706.13   |
| IEEE 300-Bus          | 706,292.32   | 706,292.32   | 706,292.32   |
| PEGASE 1354-Bus       | 1,218,096.86 | 1,218,096.86 | 1,218,096.86 |
| PEGASE 2869-Bus       | 2,386,235.33 | 2,386,235.33 | 2,386,235.33 |
| GOC 4020-Bus          | 793,634.11   | 793,634.11   | 793,634.11   |
| EPIGRIDS 5658-Bus     | 1,195,466.12 | 1,195,466.12 | 1,195,466.12 |
| EPIGRIDS 7336-Bus     | 1,855,870.94 | 1,855,870.94 | 1,855,870.94 |

<div style="text-align: left;">
  <img src="docs/source/images/dcopf_time.png" alt="DCOPF Time" width="480" height="auto">
  <p><strong>Figure:</strong> Computation time of OPF on small-scale cases.</p>
</div>

In the bar chart, the gray bar labeled "AMS Symbolic Processing" represents the time spent
on symbolic processing, while the wheat-colored bar "AMS Numeric Evaluation" represents the
time spent on system matrices calculation and optimization model construction.
The orange bar labeled "AMS GUROBI" represents the optimization-solving time using the GUROBI solver.
Similarly, the red bar labeled "AMS MOSEK" and the pink bar labeled "AMS PIQP" represent the
time used by the solvers MOSEK and PIQP, respectively.
Regarding the baselines, the blue and green bars represent the running time of MATPOWER using
solver MIPS and pandapower using solver PIPS, respectively.
The results for AMS, pandapower, and matpower are the average time consumed over ten repeat tests.

AMS is currently under active development.
Use the following resources to get involved.

-  Start from the [documentation][readthedocs] for installation and tutorial.
-  Check out examples in the [examples folder][examples]
-  Read the model verification results in the [examples/verification folder][verification]
-  Ask a question in the [GitHub Discussions][Github Discussions]
-  Report bugs or issues by submitting a [GitHub issue][GitHub issues]
-  Submit contributions using [pull requests][GitHub pull requests]
-  Read release notes highlighted [here][release notes]
-  Try in Jupyter Notebook on [Binder][Binder]
-  Check out the source code used for [benchmark][benchmark]
-  Check out and and cite our [paper][paper]

# Installation

AMS is released as ``ltbams`` on PyPI and conda-forge.
Install from PyPI using pip:

```bash
pip install ltbams
```

Install from conda-forge using conda:

```bash
conda install conda-forge::ltbams
```

Install from GitHub source:

```bash
pip install git+https://github.com/CURENT/ams.git
```

***NOTE:***
- `cvxpy` is distributed with the open source solvers CLARABEL, OSQP, and SCS, but MIP-capable solvers need separate installation
- `cvxpy` versions **below 1.5** are incompatible with `numpy` versions **2.0 and above**
- If the solver `SCIP` encounters an import error caused by a missing `libscip.9.1.dylib`, try reinstalling its Python interface by running `pip install pyscipopt --no-binary scip --force`
- `kvxopt` is recommended to install via `conda` as sometimes ``pip`` struggles to set the correct path for compiled libraries
- Versions **1.0.0** and **1.0.1** are only available on PyPI
- Version **0.9.9** has known issues and has been yanked from PyPI

# Example Usage

```python
import ams
import andes

ss = ams.load(ams.get_case('ieee14/ieee14_uced.xlsx'))

# solve RTED
ss.RTED.run(solver='CLARABEL')

ss.RTED.pg.v
>>> array([1.8743862, 0.3226138, 0.01     , 0.02     , 0.01     ])

# convert to ANDES case
sa = ss.to_andes(addfile=andes.get_case('ieee14/ieee14_full.xlsx'),
                 setup=True, verify=False)
sa
>>> <andes.system.System at 0x14bd98190>
```

# Citing AMS
If you use AMS for research or consulting, please cite the following paper in your publication that uses AMS:

> J. Wang et al., "Dynamics-incorporated Modeling Framework for Stability Constrained Scheduling Under High-penetration of Renewable Energy," in IEEE Transactions on Sustainable Energy, doi: 10.1109/TSTE.2025.3528027.

# Sponsors and Contributors
AMS is the scheduling simulation engine for the CURENT Largescale Testbed (LTB).
More information about CURENT LTB can be found at the [LTB Repository][LTB Repository].

This work was supported in part by the Engineering Research Center Program of the National Science Foundation and the Department of Energy
under NSF Award Number EEC-1041877 and the CURENT Industry Partnership Program.

This work was supported in part by the Advanced Grid Research and Development Program in the Office of Electricity at the U.S. Department of Energy.

See [GitHub contributors][GitHub contributors] for the contributor list.

# License
AMS is licensed under the [GPL v3 License](./LICENSE).

# Related Projects
- [Popular Open Source Libraries for Power System Analysis](https://github.com/jinningwang/best-of-ps)
- [G-PST Tools Portal](https://g-pst.github.io/tools/): An open tools portal with a classification approach
- [Open Source Software (OSS) for Electricity Market Research, Teaching, and Training](https://www2.econ.iastate.edu/tesfatsi/ElectricOSS.htm)

Some commercial solvers provide academic licenses, such as COPT, GUROBI, CPLEX, and MOSEK.

* * *

[GitHub releases]:       https://github.com/CURENT/ams/releases
[GitHub issues]:         https://github.com/CURENT/ams/issues
[Github Discussions]:    https://github.com/CURENT/ams/discussions
[GitHub insights]:       https://github.com/CURENT/ams/pulse
[GitHub pull requests]:  https://github.com/CURENT/ams/pulls
[GitHub contributors]:   https://github.com/CURENT/ams/graphs/contributors
[readthedocs]:           https://ams.readthedocs.io
[release notes]:         https://ams.readthedocs.io/en/latest/release-notes.html
[examples]:              https://github.com/CURENT/ams/tree/master/examples
[verification]:          https://github.com/CURENT/ams/tree/master/examples/verification
[Binder]:                https://mybinder.org/v2/gh/curent/ams/master
[LTB Repository]:        https://github.com/CURENT
[benchmark]:             https://github.com/CURENT/demo/tree/master/demo/ams_benchmark
[paper]:                 https://ieeexplore.ieee.org/document/9169830