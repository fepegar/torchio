Transforms
==========

The :py:mod:`torchio.transforms` module should remind users of
:py:mod:`torchvision.transforms`.

TorchIO transforms take as input instances of
:py:class:`~torchio.data.subject.Subject` or
:py:class:`~torchio.data.image.Image` (and its subclasses), 4D PyTorch tensors,
4D NumPy arrays, SimpleITK images or Python dictionaries
(see :py:class:`~torchio.transforms.Transform`).

For example::

   >>> import torch
   >>> import numpy as np
   >>> import torchio as tio
   >>> affine_transform = tio.RandomAffine()
   >>> tensor = torch.rand(1, 256, 256, 159)
   >>> transformed_tensor = affine_transform(tensor)
   >>> type(transformed_tensor)
   <class 'torch.Tensor'>
   >>> array = np.random.rand(1, 256, 256, 159)
   >>> transformed_array = affine_transform(array)
   >>> type(transformed_array)
   <class 'numpy.ndarray'>
   >>> subject = tio.datasets.Colin27()
   >>> transformed_subject = affine_transform(subject)
   >>> transformed_subject
   Subject(Keys: ('t1', 'head', 'brain'); images: 3)
   >>> from pprint import pprint
   >>> pprint(transformed_subject.history)  # doctest:+ELLIPSIS
   [('RandomAffine',
     {'...': array([..., ..., ...], dtype=float32),
      '...': array([..., ..., ...], dtype=float32),
      'translation': array([0., 0., 0.], dtype=float32)})]
   >>> subject.history
   []



Transforms can also be applied from the command line using
:ref:`torchio-transform`.

All transforms inherit from :py:class:`torchio.transforms.Transform`:

.. currentmodule:: torchio.transforms

.. autoclass:: Transform
   :members:

   .. automethod:: __call__



.. _Interpolation:

Interpolation
-------------

Some transforms such as
:py:class:`~torchio.transforms.RandomAffine` or
:py:class:`~torchio.transforms.RandomMotion`
need to interpolate intensity values during resampling.

The available interpolation strategies can be inferred from the elements of
:class:`torchio.transforms.interpolation.Interpolation`.

``'nearest'`` can be used for quick experimentation as it is very
fast, but produces relatively poor results.

``'linear'``, default in TorchIO, is usually a good compromise
between image quality and speed to be used for data augmentation during training.

Methods such as ``'bspline'`` or ``'lanczos'`` generate
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
