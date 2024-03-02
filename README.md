# LTB AMS

Python Software for Power System Dispatch Modeling and Co-Simulation with Dynanic, serving as the market simulator for the [CURENT Largescale Testbed][LTB Repository].

<img src="docs/source/images/sponsors/CURENT_Logo_NameOnTrans.png" alt="CURENT ERC Logo" width="300" height="auto">

|               | Latest                                                                                                                                        | Stable                                                                                                                                        |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Documentation | [![Latest Documentation](https://readthedocs.org/projects/ams/badge/?version=latest)](https://ams.readthedocs.io/en/latest/?badge=latest) | [![Documentation Status](https://readthedocs.org/projects/ams/badge/?version=stable)](https://ams.readthedocs.io/en/stable/?badge=stable) |


| Badges        |                                                                                                                                                                                                                                                     |                                                                                                                                                                                                            |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Downloads     | [![PyPI Version](https://img.shields.io/pypi/v/ltbams.svg)](https://pypi.python.org/pypi/ltbams)         | [![Conda Version](https://anaconda.org/conda-forge/ltbams/badges/version.svg)](https://anaconda.org/conda-forge/ltbams) |
| Try on Binder | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/curent/ams/master)                                                                                                                                                 |                                                                                                                                                                                                            |
| Code Quality  |[![Codacy Badge](https://app.codacy.com/project/badge/Grade/69456da1b8634f2f984bd769e35f0050)](https://app.codacy.com/gh/CURENT/ams/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)| [![Codacy Badge](https://app.codacy.com/project/badge/Coverage/69456da1b8634f2f984bd769e35f0050)](https://app.codacy.com/gh/CURENT/ams/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage) |
| Build Status  | [![GitHub Action Status](https://github.com/CURENT/ams/workflows/Python%20application/badge.svg)](https://github.com/curent/ams/actions)  | [![Build Status](https://dev.azure.com/curentltb/ams/_apis/build/status%2FCURENT.ams?branchName=master)](https://dev.azure.com/curentltb/ams/_build/latest?definitionId=2&branchName=master) |
| Structure   | [![Structure](https://img.shields.io/badge/code_base-visualize-blue)](https://mango-dune-07a8b7110.1.azurestaticapps.net/?repo=CURENT%2Fams)
| Python Version | [![Python Versions](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/) |


# Why AMS

With the built-in interface with dynamic simulation engine, ANDES, AMS enables Dynamics Interfaced Stability Constrained Production Cost and Market Operation Modeling.

AMS produces credible dispatch results and competitive performance.
The following results show the comparison of DCOPF between AMS and other tools.

| Cost [\$]       |      AMS       |  MATPOWER   | pandapower |
|----------------:|--------------:|------------:|-----------:|
| PEGASE 1354-Bus |  1,173,590.63  |  1,173,590.63 |  1,173,590.63 |
| PEGASE 2869-Bus |  2,338,915.61  |  2,338,915.61 |  2,338,915.61 |
| GOC 4020-Bus    |    793,634.11  |    793,634.11 |    793,634.11 |
| EPIGRIDS 5658-Bus| 1,195,466.12  |  1,195,466.12 |  1,195,466.12 |
| EPIGRIDS 7336-Bus| 1,855,870.94  |  1,855,870.94 |  1,855,870.94 |

<img src="docs/source/images/dcopf_time.png" alt="DCOPF Time" width="400" height="auto">

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
<!-- + Check out and and cite our [paper][arxiv paper] -->

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

# Sponsors and Contributors
AMS is the dispatch simulation engine for the CURENT Largescale Testbed (LTB).
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