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


Interpolation
-------------

Spatial transforms such as
:py:class:`~torchio.transforms.RandomAffine` or
:py:class:`~torchio.transforms.RandomElasticDeformation`
need to interpolate values during resampling.


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
