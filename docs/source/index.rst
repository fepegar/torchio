TorchIO
=======

.. image:: https://img.shields.io/pypi/dm/torchio.svg
   :target: https://pypi.org/project/torchio/
   :alt: Number of PyPI downloads

.. image:: https://badge.fury.io/py/torchio.svg
   :target: https://badge.fury.io/py/torchio
   :alt: PyPI Version

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/112NTL8uJXzcMw4PQbUvMQN-WHlVwQS3i
   :alt: Google Colab Notebook

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3598622.svg
   :target: https://doi.org/10.5281/zenodo.3598622
   :alt: DOI

.. image:: https://travis-ci.org/fepegar/torchio.svg?branch=master
   :target: https://travis-ci.org/fepegar/torchio
   :alt: Build status

.. image:: https://readthedocs.org/projects/torchio/badge/?version=latest
   :target: https://torchio.readthedocs.io/?badge=latest
   :alt: Documentation status

.. image:: https://codecov.io/gh/fepegar/torchio/branch/master/graphs/badge.svg
   :target: https://codecov.io/github/fepegar/torchio
   :alt: Coverage status

.. image:: https://img.shields.io/scrutinizer/g/fepegar/torchio.svg
   :target: https://scrutinizer-ci.com/g/fepegar/torchio/?branch=master
   :alt: Code quality


TorchIO is a Python package containing a set of tools to efficiently
read, sample and write 3D medical images in deep learning applications
written in `PyTorch <https://pytorch.org/>`_,
including intensity and spatial transforms for data augmentation and preprocessing.
Transforms include typical computer vision operations
such as random affine transformations and also domain-specific ones such as
simulation of intensity artifacts due to
`MRI magnetic field inhomogeneity <http://mriquestions.com/why-homogeneity.html>`_
or `k-space motion artifacts <http://proceedings.mlr.press/v102/shaw19a.html>`_.

This package has been greatly inspired by `NiftyNet <https://niftynet.io/>`_.


Installation
------------

.. code-block:: bash

    pip install torchio


Google Colab Jupyter Notebok
----------------------------

You can preview and run most features in TorchIO in this
`Google Colab Notebook <https://colab.research.google.com/drive/112NTL8uJXzcMw4PQbUvMQN-WHlVwQS3i>`_.


.. toctree::
   :maxdepth: 2
   :caption: API

   data/data.rst
   transforms/transforms.rst
   datasets.rst
