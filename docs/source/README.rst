#######
TorchIO
#######

|PyPI-downloads| |PyPI-version| |Google-Colab-notebook| |Docs-status| |Build-status|
|Coverage-status| |Code-Quality| |Code-Maintainability| |Slack|


TorchIO is a Python library for efficient loading, preprocessing, augmentation
and patch-based sampling of 3D medical images in deep learning,
following the design of PyTorch.

It includes multiple intensity and spatial transforms for data augmentation and preprocessing.
These transforms include typical computer vision operations
such as random affine transformations and also domain-specific ones such as
simulation of intensity artifacts due to
`MRI magnetic field inhomogeneity <http://mriquestions.com/why-homogeneity.html>`_
or `k-space motion artifacts <http://proceedings.mlr.press/v102/shaw19a.html>`_.

The code is available on `GitHub <https://github.com/fepegar/torchio>`_.


----

🎉 News: the paper is out! 🎉
*****************************

See the Credits section below for more information.

----


Credits
*******

..
  From https://stackoverflow.com/a/10766650/3956024

If you use this library for your research, please cite the paper: |paper-url|_.

.. _paper-url: https://arxiv.org/abs/2003.04696

This package has been greatly inspired by `NiftyNet <https://niftynet.io/>`_.

.. |paper-url| replace:: Pérez-García et al., 2020, *TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning*


BibTeX:

.. code-block:: latex

   @misc{fern2020torchio,
      title={TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning},
      author={Fernando Pérez-García and Rachel Sparks and Sebastien Ourselin},
      year={2020},
      eprint={2003.04696},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
   }


.. |PyPI-downloads| image:: https://img.shields.io/pypi/dm/torchio.svg?label=PyPI%20downloads&logo=python&logoColor=white
   :target: https://pypi.org/project/torchio/
   :alt: PyPI downloads

.. |PyPI-version| image:: https://img.shields.io/pypi/v/torchio?label=PyPI%20version&logo=python&logoColor=white
   :target: https://pypi.org/project/torchio/
   :alt: PyPI version

.. |Google-Colab-notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/112NTL8uJXzcMw4PQbUvMQN-WHlVwQS3i
   :alt: Google Colab notebook

.. |Docs-status| image:: https://img.shields.io/readthedocs/torchio?label=Docs&logo=Read%20the%20Docs
   :target: https://torchio.readthedocs.io/?badge=latest
   :alt: Documentation status

.. |Build-status| image:: https://img.shields.io/travis/fepegar/torchio/master.svg?label=Travis%20CI%20build&logo=travis
   :target: https://travis-ci.org/fepegar/torchio
   :alt: Build status

.. |Coverage-status| image:: https://codecov.io/gh/fepegar/torchio/branch/master/graphs/badge.svg
   :target: https://codecov.io/github/fepegar/torchio
   :alt: Coverage status

.. |Code-Quality| image:: https://img.shields.io/scrutinizer/g/fepegar/torchio.svg?label=Code%20quality&logo=scrutinizer
   :target: https://scrutinizer-ci.com/g/fepegar/torchio/?branch=master
   :alt: Code quality

.. |Slack| image:: https://img.shields.io/badge/TorchIO-Join%20on%20Slack-blueviolet?style=flat&logo=slack
   :target: https://join.slack.com/t/torchioworkspace/shared_invite/enQtOTY1NTgwNDI4NzA1LTEzMjIwZTczMGRmM2ZlMzBkZDg3YmQwY2E4OTIyYjFhZDVkZmIwOWZkNTQzYTFmYzdiNGEwZWQ4YjgwMTczZmE
   :alt: Slack

.. |Code-Maintainability| image:: https://api.codeclimate.com/v1/badges/518673e49a472dd5714d/maintainability
   :target: https://codeclimate.com/github/fepegar/torchio/maintainability
   :alt: Maintainability
