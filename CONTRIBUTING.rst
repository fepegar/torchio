.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs
`on GitHub <https://github.com/fepegar/torchio/issues/new?assignees=&labels=bug&template=bug_report.md&title=>`_.

If you are reporting a bug, please include:

* Your TorchIO version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

TorchIO could always use more documentation, whether as part of the
official TorchIO docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/fepegar/torchio/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up ``torchio`` for local development.

1) Create an issue about it on the GitHub repo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It's good practice to first discuss the proposed changes as the feature might
already be implemented.

2) Fork the ``torchio`` repo on GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3) Clone your fork locally
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone git@github.com:your_github_username_here/torchio.git
    cd torchio

4) Install your local copy into a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you use ``conda``, this is how you can set up your fork for local development::

    conda create --name torchioenv python --yes
    conda activate torchioenv
    pip install --editable .
    pip install -r requirements-dev.txt
    pre-commit install

5) Create a branch for local development using the issue number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For example, if the issue nu ber is 55::

    git checkout -b 55-name-of-your-bugfix-or-feature

Now you can make your changes locally.

6) Run unit tests
~~~~~~~~~~~~~~~~~

When you're done making changes, check that your changes pass the tests
using ``pytest``::

    pytest -x

7) Commit your changes and push your branch to GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Here's some great
advice to write good commit
messages <https://chris.beams.io/posts/git-commit>`_, and `here's some
more <https://medium.com/@joshuatauberer/write-joyous-git-commit-messages-2f98891114c4>`_)::

    git add .
    git commit -m "Fix nasty bug"
    git push origin 55-name-of-your-bugfix-or-feature

8) Check documentation
~~~~~~~~~~~~~~~~~~~~~~

If you have modified the documentation or some docstrings, build the docs and
verify that everything looks good::

    cd docs
    make html

You can also use ``livehtml`` instead, to automatically build the docs every
time you modify them and reload them in the browser::

    make livehtml

9) Submit a pull request on GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tips
----

To run a subset of tests::

    pytest tests/data/test_image.py
