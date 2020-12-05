Command-line tools
==================

``torchio-transform``
---------------------

A transform can be quickly applied to an image file using the command-line
tool ``torchio-transform``, which is automatically installed by ``pip``
during installation of TorchIO::

    $ torchio-transform input.nii.gz RandomAffine output.nii.gz --kwargs "degrees=(0,0,10) scales=0.1" --seed 42

For more information, run ``torchio-transform --help``.
