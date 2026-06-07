.. AMS documentation master file, created by
   sphinx-quickstart on Thu Jan 26 15:32:32 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==================
AMS documentation
==================


**Download documentation**: `PDF for stable version`_ | `PDF for development version`_

.. _`PDF for stable version`: https://ltb.readthedocs.io/projects/ams/downloads/en/stable/pdf/
.. _`PDF for development version`: https://ltb.readthedocs.io/projects/ams/downloads/en/latest/pdf/


**Useful Links**: `Binary Installer`_ | `Source Repository`_ | `Report Issues`_
| `Q&A`_ | `Try in Jupyter Notebooks`_ | `LTB Repository`_ | `ANDES Repository`_

.. _`Binary Installer`: https://pypi.org/project/ltbams/
.. _`Source Repository`: https://github.com/CURENT/ams
.. _`Report Issues`: https://github.com/CURENT/ams/issues
.. _`Q&A`: https://github.com/CURENT/ams/discussions
.. _`Try in Jupyter Notebooks`: https://mybinder.org/v2/gh/curent/ams/master
.. _`ANDES Repository`: https://github.com/CURENT/andes
.. _`LTB Repository`: https://github.com/CURENT/

.. image:: /images/sponsors/CURENT_Logo_NameOnTrans.png
   :alt: CURENT Logo
   :width: 300px
   :height: 74.2px

LTB AMS is an open-source Python library for power system scheduling modeling and
co-simulation with dynamics, serving as the market simulator for the CURENT Large
scale Testbed (LTB). It implements a descriptive modeling framework for scheduling
problems, solved via CVXPY with third-party solvers. AMS enables tight
interoperability with the dynamic simulator ANDES for stability-constrained scheduling
studies.

.. panels::
    :card: + intro-card text-center
    :column: col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex

    ---

    Getting started
    ^^^^^^^^^^^^^^^

    New to AMS? Check out the Getting Started guides. They contain an introduction
    to the AMS command-line interface, scripting usages, as well as guides to
    configure AMS and work with case files.

    +++

    .. link-button:: getting-started
            :type: ref
            :text: To the getting started guides
            :classes: btn-block btn-secondary stretched-link

    ---

    Tutorials
    ^^^^^^^^^

    The tutorials provide in-depth usage of AMS in a Python scripting environment.
    Scheduling studies and co-simulation with ANDES dynamics are shown with
    explanation.

    +++

    .. link-button:: tutorials
            :type: ref
            :text: To the tutorials
            :classes: btn-block btn-secondary stretched-link

    ---

    Modeling guide
    ^^^^^^^^^^^^^^

    Looking to implement new scheduling formulations in AMS? The modeling guide
    provides in-depth information on the design philosophy, data structure, and
    implementation of the scheduling modeling framework.

    +++

    .. link-button:: modeling
            :type: ref
            :text: To the modeling guide
            :classes: btn-block btn-secondary stretched-link
    ---

    Reference
    ^^^^^^^^^

    The reference contains a detailed description of the AMS routines, models, and
    package API. It describes how the methods work and which parameters can be used.
    It assumes that you have an understanding of the key concepts.

    +++

    .. link-button:: reference
            :type: ref
            :text: To the reference
            :classes: btn-block btn-secondary stretched-link

    ---
    :column: col-12 p-3

    Using AMS for Research?
    ^^^^^^^^^^^^^^^^^^^^^^^^^
    Please cite our paper [Wang2025]_ if AMS is used in your research for
    publication.


.. [Wang2025] J. Wang et al., "Dynamics-Incorporated Modeling Framework for Stability
       Constrained Scheduling Under High-Penetration of Renewable Energy," in IEEE
       Transactions on Sustainable Energy,  vol. 16, no. 3, pp. 1673-1685, July 2025,
       doi: 10.1109/TSTE.2025.3528027.


.. toctree::
   :maxdepth: 3
   :caption: AMS Manual
   :hidden:

   getting_started/index
   examples/index
   modeling/index
   release-notes
   reference/index
