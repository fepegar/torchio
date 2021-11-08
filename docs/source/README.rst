#######
TorchIO
#######

|PyPI-downloads| |PyPI-version| |Conda-version| |Google-Colab-notebook| |Docs-status|
|Tests-status| |Lint-status|
|Coverage-codecov| |Code-Quality| |Code-Maintainability| |pre-commit|
|Slack| |Twitter| |Twitter-commits| |YouTube|


TorchIO is a Python library for efficient loading, preprocessing, augmentation
and patch-based sampling of 3D medical images in deep learning,
following the design of PyTorch.

It includes multiple intensity and spatial transforms for data augmentation and
preprocessing.
These transforms include typical computer vision operations
such as random affine transformations and also domain-specific ones such as
simulation of intensity artifacts due to
`MRI magnetic field inhomogeneity (bias) <https://mriquestions.com/why-homogeneity.html>`_
or `k-space motion artifacts <http://proceedings.mlr.press/v102/shaw19a.html>`_.

TorchIO is part of the official `PyTorch Ecosystem <https://pytorch.org/ecosystem/>`_,
and it was featured at the `PyTorch Ecosystem Day 2021 <https://pytorch.org/ecosystem/pted/2021>`_.

Many groups have used TorchIO for their research.
The complete list of citations is available on `Google Scholar <https://scholar.google.co.uk/scholar?cites=8711392719159421861&sciodt=0,5&hl=en>`_.

The code is available on `GitHub <https://github.com/fepegar/torchio>`_.
If you like TorchIO, please give the repo a star!

.. raw:: html

   <a class="github-button" href="https://github.com/fepegar/torchio" data-icon="octicon-star" data-show-count="true" aria-label="Star fepegar/torchio on GitHub">Star</a>
   <script async defer src="https://buttons.github.io/buttons.js"></script>

See :doc:`Getting started <quickstart>` for installation instructions and a
usage overview.

Contributions are more than welcome.
Please check our `contributing guide <https://github.com/fepegar/torchio/blob/main/CONTRIBUTING.rst>`_
if you would like to contribute.

If you have questions, feel free to ask in the discussions tab:

.. raw:: html

   <a class="github-button" href="https://github.com/fepegar/torchio/discussions" data-icon="octicon-comment-discussion" aria-label="Discuss fepegar/torchio on GitHub">Discuss</a>
   <script async defer src="https://buttons.github.io/buttons.js"></script>

If you found a bug or have a feature request, please open an issue:

.. raw:: html

   <!-- Place this tag where you want the button to render. -->
   <a class="github-button" href="https://github.com/fepegar/torchio/issues" data-icon="octicon-issue-opened" data-show-count="true" aria-label="Issue fepegar/torchio on GitHub">Issue</a>
   <!-- Place this tag in your head or just before your close body tag. -->
   <script async defer src="https://buttons.github.io/buttons.js"></script>


Credits
*******

..
  From https://stackoverflow.com/a/10766650/3956024

If you use this library for your research,
please cite our paper:

`F. Pérez-García, R. Sparks, and S. Ourselin. TorchIO: a Python library for
efficient loading, preprocessing, augmentation and patch-based sampling of
medical images in deep learning. Computer Methods and Programs in Biomedicine
(June 2021), p. 106236. ISSN:
0169-2607.doi:10.1016/j.cmpb.2021.106236. <https://doi.org/10.1016/j.cmpb.2021.106236>`_


BibTeX:

.. code-block:: latex

   @article{perez-garcia_torchio_2021,
      title = {TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning},
      journal = {Computer Methods and Programs in Biomedicine},
      pages = {106236},
      year = {2021},
      issn = {0169-2607},
      doi = {https://doi.org/10.1016/j.cmpb.2021.106236},
      url = {https://www.sciencedirect.com/science/article/pii/S0169260721003102},
      author = {P{\'e}rez-Garc{\'i}a, Fernando and Sparks, Rachel and Ourselin, S{\'e}bastien},
   }

This project is supported by the following institutions:

* `Engineering and Physical Sciences Research Council (EPSRC) & UK Research and Innovation (UKRI) <https://epsrc.ukri.org/>`_
* `EPSRC Centre for Doctoral Training in Intelligent, Integrated Imaging In Healthcare (i4health) <https://www.ucl.ac.uk/intelligent-imaging-healthcare/>`_ (University College London)
* `Wellcome / EPSRC Centre for Interventional and Surgical Sciences (WEISS) <https://www.ucl.ac.uk/interventional-surgical-sciences/>`_ (University College London)
* `School of Biomedical Engineering & Imaging Sciences (BMEIS) <https://www.kcl.ac.uk/bmeis>`_ (King's College London)

This library has been greatly inspired by
`NiftyNet <https://github.com/NifTK/NiftyNet>`_ which is no longer maintained.


.. |PyPI-downloads| image:: https://img.shields.io/pypi/dm/torchio.svg?label=PyPI%20downloads&logo=python&logoColor=white
   :target: https://pypi.org/project/torchio/
   :alt: PyPI downloads

.. |PyPI-version| image:: https://img.shields.io/pypi/v/torchio?label=PyPI%20version&logo=python&logoColor=white
   :target: https://pypi.org/project/torchio/
   :alt: PyPI version

.. |Conda-version| image:: https://img.shields.io/conda/v/conda-forge/torchio.svg?label=conda-forge&logo=conda-forge
   :target: https://anaconda.org/conda-forge/torchio
   :alt: Conda version

.. |Google-Colab-notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://github.com/fepegar/torchio/blob/main/tutorials/README.md
   :alt: Google Colab notebooks

.. |Docs-status| image:: https://img.shields.io/readthedocs/torchio?label=Docs&logo=Read%20the%20Docs
   :target: https://torchio.rtfd.io/?badge=latest
   :alt: Documentation status

.. |Lint-status| image:: https://github.com/fepegar/torchio/actions/workflows/lint.yml/badge.svg
   :target: https://github.com/fepegar/torchio/actions/workflows/lint.yml
   :alt: Lint status

.. |Tests-status| image:: https://github.com/fepegar/torchio/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/fepegar/torchio/actions/workflows/tests.yml
   :alt: Tests status

.. |Coverage-codecov| image:: https://codecov.io/gh/fepegar/torchio/branch/main/graphs/badge.svg
   :target: https://codecov.io/github/fepegar/torchio
   :alt: Coverage status

.. |Code-Quality| image:: https://img.shields.io/scrutinizer/g/fepegar/torchio.svg?label=Code%20quality&logo=scrutinizer
   :target: https://scrutinizer-ci.com/g/fepegar/torchio/?branch=main
   :alt: Code quality

.. |Slack| image:: https://img.shields.io/badge/TorchIO-Join%20on%20Slack-blueviolet?style=flat&logo=slack
   :target: https://join.slack.com/t/torchioworkspace/shared_invite/zt-exgpd5rm-BTpxg2MazwiiMDw7X9xMFg
   :alt: Slack

.. |YouTube| image:: https://img.shields.io/youtube/views/UEUVSw5-M9M?label=watch&style=social
   :target: https://www.youtube.com/watch?v=UEUVSw5-M9M
   :alt: YouTube

.. |Twitter| image:: https://img.shields.io/twitter/url/https/twitter.com/TorchIOLib.svg?style=social&label=Follow%20%40TorchIOLib
   :target: https://twitter.com/TorchIOLib
   :alt: Twitter

.. |Twitter-commits| image:: https://img.shields.io/twitter/url/https/twitter.com/TorchIO_commits.svg?style=social&label=Follow%20%40TorchIO_commits
   :target: https://twitter.com/TorchIO_commits
   :alt: Twitter commits

.. |Code-Maintainability| image:: https://api.codeclimate.com/v1/badges/518673e49a472dd5714d/maintainability
   :target: https://codeclimate.com/github/fepegar/torchio/maintainability
   :alt: Maintainability

.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
