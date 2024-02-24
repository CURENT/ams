.. AMS documentation master file, created by
   sphinx-quickstart on Thu Jan 26 15:32:32 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==================
AMS documentation
==================
**Useful Links**: `Source Repository`_ | `Report Issues`_ | `Q&A`_  | `LTB Repository`_
| `ANDES Repository`_

.. _`Source Repository`: https://github.com/jinningwang/ams
.. _`Report Issues`: https://github.com/jinningwang/ams/issues
.. _`Q&A`: https://github.com/jinningwang/ams/discussions
.. _`ANDES Repository`: https://github.com/CURENT/andes
.. _`LTB Repository`: https://github.com/CURENT/

LTB AMS is an open-source packages for dispatch modeling, serving as the market
simulator for the CURENT Large scale Testbed (LTB).

AMS enables **flexible** dispatch modeling and **interoprability** with the in-house
dynamic simulator ANDES.

.. panels::
    :card: + intro-card text-center
    :column: col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex

    ---

    Getting started
    ^^^^^^^^^^^^^^^

    New to AMS? Check out the getting started guides.

    +++

    .. link-button:: getting-started
            :type: ref
            :text: To the getting started guides
            :classes: btn-block btn-secondary stretched-link

    ---

    Examples
    ^^^^^^^^

    The examples of using AMS for power system dispatch study.

    +++

    .. link-button:: scripting_examples
            :type: ref
            :text: To the examples
            :classes: btn-block btn-secondary stretched-link

    ---

    Model development guide
    ^^^^^^^^^^^^^^^^^^^^^^^

    New dispatch modeling in AMS.

    +++

    .. link-button:: development
            :type: ref
            :text: To the development guide
            :classes: btn-block btn-secondary stretched-link
    ---

    API reference
    ^^^^^^^^^^^^^

    The API reference of AMS.

    +++

    .. link-button:: api_reference
            :type: ref
            :text: To the API reference
            :classes: btn-block btn-secondary stretched-link

    ---
    :column: col-12 p-3

    Using AMS for Research?
    ^^^^^^^^^^^^^^^^^^^^^^^^^
    Please cite our paper [Cui2021]_ if AMS is used in your research for
    publication.


.. [Cui2021] H. Cui, F. Li and K. Tomsovic, "Hybrid Symbolic-Numeric Framework
       for Power System Modeling and Analysis," in IEEE Transactions on Power
       Systems, vol. 36, no. 2, pp. 1373-1384, March 2021, doi:
       10.1109/TPWRS.2020.3017019.


.. toctree::
   :maxdepth: 3
   :caption: AMS Manual
   :hidden:

   getting_started/index
   examples/index
   modeling/index
   release-notes
   routineref
   modelref
   api