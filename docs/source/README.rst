#######
TorchIO
#######

|PyPI-downloads| |PyPI-version| |Conda-version| |Google-Colab-notebook|
|Docs-status| |Tests-status|
|Coverage-codecov| |Code-Quality| |Code-Maintainability| |pre-commit|
|Slack| |Twitter| |Twitter-commits| |YouTube|


TorchIO is an open-source Python library for efficient loading, preprocessing,
augmentation and patch-based sampling of 3D medical images in deep learning,
following the design of PyTorch.

It includes multiple intensity and spatial transforms for data augmentation and
preprocessing.
These transforms include typical computer vision operations
such as random affine transformations and also domain-specific ones such as
simulation of intensity artifacts due to
`MRI magnetic field inhomogeneity (bias) <https://mriquestions.com/why-homogeneity.html>`_
or `k-space motion artifacts <http://proceedings.mlr.press/v102/shaw19a.html>`_.

TorchIO is part of the official `PyTorch Ecosystem <https://pytorch.org/ecosystem/>`_,
and was featured at
the `PyTorch Ecosystem Day 2021 <https://pytorch.org/ecosystem/pted/2021>`_ and
the `PyTorch Developer Day 2021 <https://pytorch.org/blog/pytorch-developer-day-2021>`_.

Many groups have used TorchIO for their research.
The complete list of citations is available on `Google Scholar <https://scholar.google.co.uk/scholar?cites=8711392719159421861&sciodt=0,5&hl=en>`_, and the
`dependents list <https://github.com/fepegar/torchio/network/dependents>`_ is
available on GitHub.

The code is available on `GitHub <https://github.com/fepegar/torchio>`_.
If you like TorchIO, please go to the repository and star it!

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


.. include:: credits.rst


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
