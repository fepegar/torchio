Command-line tools
******************

.. _torchio-transform:

``torchio-transform``
=====================

A transform can be quickly applied to an image file using the command-line
tool ``torchio-transform``, which is automatically installed by PIP when
during installation of TorchIO::

    $ torchio-transform input.nii.gz RandomMotion output.nii.gz --kwargs "num_transforms=4" --seed 42

For more information, run ``torchio-transform --help``.
