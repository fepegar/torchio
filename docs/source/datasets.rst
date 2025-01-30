Medical image datasets
======================

TorchIO offers tools to easily download publicly available datasets from
different institutions and modalities.

The interface is similar to :mod:`torchvision.datasets`.

If you use any of them, please visit the corresponding website (linked in each
description) and make sure you comply with any data usage agreement and you
acknowledge the corresponding authors' publications.

If you would like to add a dataset here, please open a discussion on the
GitHub repository:

.. raw:: html

   <a class="github-button" href="https://github.com/TorchIO-project/torchio/discussions" data-icon="octicon-comment-discussion" aria-label="Discuss TorchIO-project/torchio on GitHub">Discuss</a>
   <script async defer src="https://buttons.github.io/buttons.js"></script>


IXI
---

.. automodule:: torchio.datasets.ixi
.. currentmodule:: torchio.datasets.ixi

:class:`IXI`
~~~~~~~~~~~~~

.. autoclass:: IXI

:class:`IXITiny`
~~~~~~~~~~~~~~~~~

.. autoclass:: IXITiny

EPISURG
-------

.. currentmodule:: torchio.datasets.episurg

:class:`EPISURG`
~~~~~~~~~~~~~~~~

.. autoclass:: EPISURG
    :members:

Kaggle datasets
---------------

.. currentmodule:: torchio.datasets.rsna_miccai

:class:`RSNAMICCAI`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: RSNAMICCAI

.. currentmodule:: torchio.datasets.rsna_spine_fracture

:class:`RSNACervicalSpineFracture`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RSNACervicalSpineFracture

MNI
---

.. automodule:: torchio.datasets.mni
.. currentmodule:: torchio.datasets.mni


:class:`ICBM2009CNonlinearSymmetric`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ICBM2009CNonlinearSymmetric

:class:`Colin27`
~~~~~~~~~~~~~~~~

.. autoclass:: Colin27
.. plot::

    import torchio as tio
    subject = tio.datasets.Colin27()
    subject.plot()


:class:`Pediatric`
~~~~~~~~~~~~~~~~~~

.. autoclass:: Pediatric
.. plot::

    import torchio as tio
    subject = tio.datasets.Pediatric((4.5, 8.5))
    subject.plot()


:class:`Sheep`
~~~~~~~~~~~~~~

.. autoclass:: Sheep
.. plot::

    import torchio as tio
    subject = tio.datasets.Sheep()
    subject.plot()


.. currentmodule:: torchio.datasets.bite

:class:`BITE3`
~~~~~~~~~~~~~~

.. autoclass:: BITE3


ITK-SNAP
--------

.. automodule:: torchio.datasets.itk_snap
.. currentmodule:: torchio.datasets.itk_snap


:class:`BrainTumor`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: BrainTumor
.. plot::

    import torchio as tio
    tio.datasets.BrainTumor().plot()


:class:`T1T2`
~~~~~~~~~~~~~

.. autoclass:: T1T2
.. plot::

    import torchio as tio
    subject = tio.datasets.T1T2()
    subject.plot()


:class:`AorticValve`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AorticValve
.. plot::

    import torchio as tio
    subject = tio.datasets.AorticValve()
    subject.plot()


3D Slicer
---------

.. automodule:: torchio.datasets.slicer
.. currentmodule:: torchio.datasets.slicer

:class:`Slicer`
~~~~~~~~~~~~~~~

.. autoclass:: Slicer
.. plot::

    import torchio as tio
    subject = tio.datasets.Slicer()
    subject.plot()


FPG
---

.. currentmodule:: torchio.datasets.fpg

.. autoclass:: FPG
.. plot::

    import torchio as tio
    subject = tio.datasets.FPG()
    subject.plot()

.. plot::

    import torchio as tio
    subject = tio.datasets.FPG(load_all=True)
    subject.plot()


MedMNIST
--------

.. currentmodule:: torchio.datasets.medmnist


.. autoclass:: OrganMNIST3D

.. plot::

    import torch
    import torchio as tio
    from einops import rearrange
    rows, cols = 16, 28
    dataset = tio.datasets.OrganMNIST3D('train')
    loader = tio.SubjectsLoader(dataset, batch_size=rows * cols)
    batch = tio.utils.get_first_item(loader)
    tensor = batch['image'][tio.DATA]
    pattern = '(b1 b2) c x y z -> c x (b1 y) (b2 z)'
    tensor = rearrange(tensor, pattern, b1=rows, b2=cols)
    sx = tensor.shape[1]
    plt.imshow(tensor[0, sx // 2], cmap='gray')


.. autoclass:: NoduleMNIST3D

.. plot::

    import torch
    import torchio as tio
    from einops import rearrange
    rows, cols = 16, 28
    dataset = tio.datasets.NoduleMNIST3D('train')
    loader = tio.SubjectsLoader(dataset, batch_size=rows * cols)
    batch = tio.utils.get_first_item(loader)
    tensor = batch['image'][tio.DATA]
    pattern = '(b1 b2) c x y z -> c x (b1 y) (b2 z)'
    tensor = rearrange(tensor, pattern, b1=rows, b2=cols)
    sx = tensor.shape[1]
    plt.imshow(tensor[0, sx // 2], cmap='gray')


.. autoclass:: AdrenalMNIST3D

.. plot::

    import torch
    import torchio as tio
    from einops import rearrange
    rows, cols = 16, 28
    dataset = tio.datasets.AdrenalMNIST3D('train')
    loader = tio.SubjectsLoader(dataset, batch_size=rows * cols)
    batch = tio.utils.get_first_item(loader)
    tensor = batch['image'][tio.DATA]
    pattern = '(b1 b2) c x y z -> c x (b1 y) (b2 z)'
    tensor = rearrange(tensor, pattern, b1=rows, b2=cols)
    sx = tensor.shape[1]
    plt.imshow(tensor[0, sx // 2], cmap='gray')


.. autoclass:: FractureMNIST3D

.. plot::

    import torch
    import torchio as tio
    from einops import rearrange
    rows, cols = 16, 28
    dataset = tio.datasets.FractureMNIST3D('train')
    loader = tio.SubjectsLoader(dataset, batch_size=rows * cols)
    batch = tio.utils.get_first_item(loader)
    tensor = batch['image'][tio.DATA]
    pattern = '(b1 b2) c x y z -> c x (b1 y) (b2 z)'
    tensor = rearrange(tensor, pattern, b1=rows, b2=cols)
    sx = tensor.shape[1]
    plt.imshow(tensor[0, sx // 2], cmap='gray')


.. autoclass:: VesselMNIST3D

.. plot::

    import torch
    import torchio as tio
    from einops import rearrange
    rows, cols = 16, 28
    dataset = tio.datasets.VesselMNIST3D('train')
    loader = tio.SubjectsLoader(dataset, batch_size=rows * cols)
    batch = tio.utils.get_first_item(loader)
    tensor = batch['image'][tio.DATA]
    pattern = '(b1 b2) c x y z -> c x (b1 y) (b2 z)'
    tensor = rearrange(tensor, pattern, b1=rows, b2=cols)
    sx = tensor.shape[1]
    plt.imshow(tensor[0, sx // 2], cmap='gray')


.. autoclass:: SynapseMNIST3D

.. plot::

    import torch
    import torchio as tio
    from einops import rearrange
    rows, cols = 16, 28
    dataset = tio.datasets.SynapseMNIST3D('train')
    loader = tio.SubjectsLoader(dataset, batch_size=rows * cols)
    batch = tio.utils.get_first_item(loader)
    tensor = batch['image'][tio.DATA]
    pattern = '(b1 b2) c x y z -> c x (b1 y) (b2 z)'
    tensor = rearrange(tensor, pattern, b1=rows, b2=cols)
    sx = tensor.shape[1]
    plt.imshow(tensor[0, sx // 2], cmap='gray')
