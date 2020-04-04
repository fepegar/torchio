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

Ready to contribute? Here's how to set up `torchio` for local development.

1. Create an issue about it on the GitHub repo.
2. Fork the `torchio` repo on GitHub.
3. Clone your fork locally::

    $ git clone git@github.com:your_name_here/torchio.git
    $ cd torchio

4. Install your local copy into a virtual environment.
If you use ``conda``, this is how you can set up your fork for local development::

    $ conda create --name torchioenv python --yes
    $ conda activate torchioenv
    $ python setup.py develop

5. Create a branch for local development using the issue number. If the issue
is #55::

    $ git checkout -b 55-name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the
   tests, including testing other Python versions with tox::

    $ flake8 torchio tests
    $ pytest
    $ tox

   To get ``flake8`` and ``tox``, just ``pip install`` them into your virtual environment.

6. Commit your changes and push your branch to GitHub (`here's some great
advice to write good commit
messages <https://chris.beams.io/posts/git-commit>`_, and `here's some
more <https://medium.com/@joshuatauberer/write-joyous-git-commit-messages-2f98891114c4>`_)::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin 55-name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. You can
   put your new functionality into a function with a docstring.
3. The pull request should work for Python 3.6 and 3.7. Check
   https://travis-ci.org/fepegar/torchio/
   and make sure that the tests pass for all supported Python versions.

Tips
----

To run a subset of tests::


    $ python -m unittest tests.test_torchio
