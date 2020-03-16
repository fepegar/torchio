Command-line tools
******************

``torchio-transform``
=====================

A transform can be quickly applied to an image file using the command-line
tool ``torchio-transform``::

    $ torchio-transform input.nii.gz RandomMotion output.nii.gz --kwargs "proportion_to_augment=1 num_transforms=4"
