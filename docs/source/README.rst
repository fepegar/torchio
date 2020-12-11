#######
TorchIO
#######

|PyPI-downloads| |PyPI-version| |Google-Colab-notebook| |Docs-status|
|Build-status| |Coverage-codecov| |Coverage-coveralls| |Code-Quality|
|Code-Maintainability| |pre-commit| |Slack|


TorchIO is a Python library for efficient loading, preprocessing, augmentation
and patch-based sampling of 3D medical images in deep learning,
following the design of PyTorch.

It includes multiple intensity and spatial transforms for data augmentation and
preprocessing.
These transforms include typical computer vision operations
such as random affine transformations and also domain-specific ones such as
simulation of intensity artifacts due to
`MRI magnetic field inhomogeneity (bias) <http://mriquestions.com/why-homogeneity.html>`_
or `k-space motion artifacts <http://proceedings.mlr.press/v102/shaw19a.html>`_.

The code is available on `GitHub <https://github.com/fepegar/torchio>`_.

See :doc:`Getting started <quickstart>` for installation instructions and a
usage overview.


Credits
*******

..
  From https://stackoverflow.com/a/10766650/3956024

If you use this library for your research,
please cite our preprint:

|paper-url|_

.. _paper-url: https://arxiv.org/abs/2003.04696

.. |paper-url| replace:: Pérez-García, F., Sparks, R., Ourselin, S.: TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning. arXiv:2003.04696 [cs, eess, stat] (Mar 2020), http://arxiv.org/abs/2003.04696, arXiv: 2003.04696


BibTeX:

.. code-block:: latex

   @article{perez-garcia_torchio_2020,
      title = {{TorchIO}: a {Python} library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning},
      shorttitle = {{TorchIO}},
      url = {http://arxiv.org/abs/2003.04696},
      urldate = {2020-03-11},
      journal = {arXiv:2003.04696 [cs, eess, stat]},
      author = {P{\'e}rez-Garc{\'i}a, Fernando and Sparks, Rachel and Ourselin, Sebastien},
      month = mar,
      year = {2020},
      note = {arXiv: 2003.04696},
      keywords = {Computer Science - Computer Vision and Pattern Recognition, Electrical Engineering and Systems Science - Image and Video Processing, Computer Science - Machine Learning, Computer Science - Artificial Intelligence, Statistics - Machine Learning},
   }

This project is supported by the
`Wellcome / EPSRC Centre for Interventional and Surgical Sciences
(WEISS) <https://www.ucl.ac.uk/interventional-surgical-sciences/>`_
(University College London) and the
`School of Biomedical Engineering & Imaging Sciences
(BMEIS) <https://www.kcl.ac.uk/bmeis>`_
(King's College London).

.. image:: ../images/weiss.jpg
    :width: 300
    :alt: Wellcome / EPSRC Centre for Interventional and Surgical Sciences


.. image:: ../images/cme.svg
    :width: 250
    :alt: School of Biomedical Engineering & Imaging Sciences

This package has been greatly inspired by `NiftyNet <https://niftynet.io/>`_.


.. |PyPI-downloads| image:: https://img.shields.io/pypi/dm/torchio.svg?label=PyPI%20downloads&logo=python&logoColor=white
   :target: https://pypi.org/project/torchio/
   :alt: PyPI downloads

.. |PyPI-version| image:: https://img.shields.io/pypi/v/torchio?label=PyPI%20version&logo=python&logoColor=white
   :target: https://pypi.org/project/torchio/
   :alt: PyPI version

.. |Google-Colab-notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://github.com/fepegar/torchio/blob/master/examples/README.md
   :alt: Google Colab notebooks

.. |Docs-status| image:: https://img.shields.io/readthedocs/torchio?label=Docs&logo=Read%20the%20Docs
   :target: http://torchio.rtfd.io/?badge=latest
   :alt: Documentation status

.. |Build-status| image:: https://img.shields.io/travis/fepegar/torchio/master.svg?label=Travis%20CI%20build&logo=travis
   :target: https://travis-ci.org/fepegar/torchio
   :alt: Build status

.. |Coverage-codecov| image:: https://codecov.io/gh/fepegar/torchio/branch/master/graphs/badge.svg
   :target: https://codecov.io/github/fepegar/torchio
   :alt: Coverage status

.. |Coverage-coveralls| image:: https://coveralls.io/repos/github/fepegar/torchio/badge.svg?branch=master
   :target: https://coveralls.io/github/fepegar/torchio?branch=master
   :alt: Coverage status

.. |Code-Quality| image:: https://img.shields.io/scrutinizer/g/fepegar/torchio.svg?label=Code%20quality&logo=scrutinizer
   :target: https://scrutinizer-ci.com/g/fepegar/torchio/?branch=master
   :alt: Code quality

.. |Slack| image:: https://img.shields.io/badge/TorchIO-Join%20on%20Slack-blueviolet?style=flat&logo=slack
   :target: https://join.slack.com/t/torchioworkspace/shared_invite/zt-exgpd5rm-BTpxg2MazwiiMDw7X9xMFg
   :alt: Slack

.. |Code-Maintainability| image:: https://api.codeclimate.com/v1/badges/518673e49a472dd5714d/maintainability
   :target: https://codeclimate.com/github/fepegar/torchio/maintainability
   :alt: Maintainability

.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
