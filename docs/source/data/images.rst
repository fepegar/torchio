Images dataset
==============

The :py:class:`~torchio.data.images.ImagesDataset` class
is one of the most important in TorchIO.
It inherits directly from :class:`torch.utils.data.Dataset`.
Here is a usage example:

>>> import torchio
>>> from torchio import ImagesDataset, Image, Subject
>>> from torchio.transforms import RescaleIntensity, RandomAffine
>>> from torchvision.transforms import Compose
>>> subject_a = Subject([
...     t1=Image('~/Dropbox/MRI/t1.nrrd', torchio.INTENSITY),
...     label=Image('~/Dropbox/MRI/t1_seg.nii.gz', torchio.LABEL),
>>> ])
>>> subject_b = Subject(
...     t1=Image('/tmp/colin27_t1_tal_lin.nii.gz', torchio.INTENSITY),
...     t2=Image('/tmp/colin27_t2_tal_lin.nii', torchio.INTENSITY),
...     label=Image('/tmp/colin27_seg1.nii.gz', torchio.LABEL),
... )
>>> subjects_list = [subject_a, subject_b]
>>> transforms = [
...     RescaleIntensity((0, 1)),
...     RandomAffine(),
... ]
>>> transform = Compose(transforms)
>>> subjects_dataset = ImagesDataset(subjects_list, transform=transform)
>>> subject_sample = subjects_dataset[0]


.. currentmodule:: torchio.data.images

:class:`ImagesDataset`
----------------------

.. autoclass:: ImagesDataset
    :members:
    :show-inheritance:


:class:`Subject`
----------------

.. autoclass:: Subject
    :members:
    :show-inheritance:


:class:`Image`
--------------

.. autoclass:: Image
