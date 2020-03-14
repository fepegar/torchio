***************
Getting started
***************

Installation
============

.. code-block:: bash

    $ pip install torchio


Hello World
===========

This example shows the basic usage of TorchIO, where an
:py:class:`~torchio.data.images.ImagesDataset` is passed to
a PyTorch :py:class:`~torch.utils.data.DataLoader` to generate training batches
of 3D images that are loaded, preprocessed and augmented in on the fly,
in parallel::

    import torchio
    from torchio.transforms import Rescale, RandomAffine
    from torchvision.transforms import Compose
    from torch.utils.data import DataLoader
    subject_a = torchio.Subject([
        torchio.Image('t1', 'subject_a.nii.gz', torchio.INTENSITY),
        torchio.Image('label', 'subject_a.nii', torchio.LABEL),
    ])
    subject_b = torchio.Subject(
        torchio.Image('t1', 'subject_b_dicom_folder', torchio.INTENSITY),
        torchio.Image('label', 'subject_b_seg.nrrd', torchio.LABEL),
    )
    subjects_list = [subject_a, subject_b]
    transforms = [
        Rescale((0, 1)),  # applied only to torchio.INTENSITY images
        RandomAffine(),  # applied to all images in the sample
    ]  # Some preprocessing and augmentation
    transform = Compose(transforms)
    subjects_dataset = torchio.ImagesDataset(subjects_list, transform=transform)
    training_loader = DataLoader(subjects_dataset, batch_size=4, num_workers=4)
    for subjects_batch in training_loader:
        inputs = subjects_batch['t1'][torchio.DATA]
        target = subjects_batch['label'][torchio.DATA]




Google Colab Jupyter Notebok
============================

|Google-Colab-notebook|

You can preview and run most features in TorchIO in this
`Google Colab Notebook <https://colab.research.google.com/drive/112NTL8uJXzcMw4PQbUvMQN-WHlVwQS3i>`_.

.. |Google-Colab-notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/112NTL8uJXzcMw4PQbUvMQN-WHlVwQS3i
   :alt: Google Colab notebook
