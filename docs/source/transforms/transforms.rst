Transforms
==========

The :py:mod:`torchio.transforms` module should remind users of
:py:mod:`torchvision.transforms`.

TorchIO transforms take as input samples generated
by an :py:class:`~torchio.data.image.ImagesDataset`.

For example::

   >>> from torchio.transforms import RandomAffine
   >>> from torchio.datasets import IXITiny
   >>> affine_transform = RandomAffine()
   >>> dataset = IXITiny('ixi', download=True)
   >>> sample = dataset[0]
   >>> transformed_sample = affine_transform(sample)


All transforms inherit from :py:class:`torchio.transforms.Transform`:


.. currentmodule:: torchio.transforms

.. autoclass:: Transform



Command-line interface
----------------------

A transform can be quickly applied to an image file using the command-line
tool ``torchio-transform``::

   $ torchio-transform input.nii.gz RandomMotion output.nii.gz --kwargs "num_transforms=4 --seed 42"

For more information, run ``torchio-transform --help``.


.. _Interpolation:

Interpolation
-------------

Some transforms such as
:py:class:`~torchio.transforms.RandomAffine` or
:py:class:`~torchio.transforms.RandomMotion`
need to interpolate intensity values during resampling.

The available interpolation strategies are enumerated in
:class:`torchio.transforms.interpolation.Interpolation`.

``Interpolation.NEAREST`` can be used for quick experimentation as it is very
fast, but produces relatively poor results.

``Interpolation.LINEAR``, defaut in TorchIO, is usually a good compromise
between image quality and speed to be used for data augmentation during training.

Methods such as ``Interpolation.BSPLINE`` or ``Interpolation.LANCZOS`` generate
high-quality results, but are generally slower. They can be used to obtain
optimal resampling results during offline data preprocessing.

Visit the
`ITK docs <https://itk.org/Doxygen/html/group__ImageInterpolators.html>`_
for more information and see
`this SimpleITK example <https://simpleitk-prototype.readthedocs.io/en/latest/user_guide/transforms/plot_interpolation.html>`_
for some interpolation results on test images.


.. autoclass:: Interpolation
   :show-inheritance:
   :members:
   :undoc-members:




Transforms API
--------------

.. toctree::
   :maxdepth: 3

   preprocessing.rst
   augmentation.rst
   others.rst
