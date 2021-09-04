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

   <a class="github-button" href="https://github.com/fepegar/torchio/discussions" data-icon="octicon-comment-discussion" aria-label="Discuss fepegar/torchio on GitHub">Discuss</a>
   <script async defer src="https://buttons.github.io/buttons.js"></script>


IXI
---

.. automodule:: torchio.datasets.ixi
.. currentmodule:: torchio.datasets.ixi

:class:`IXI`
~~~~~~~~~~~~~

.. autoclass:: IXI
    :members:
    :show-inheritance:


:class:`IXITiny`
~~~~~~~~~~~~~~~~~

.. autoclass:: IXITiny
    :members:
    :show-inheritance:


EPISURG
-------

.. currentmodule:: torchio.datasets.episurg

:class:`EPISURG`
~~~~~~~~~~~~~~~~

.. autoclass:: EPISURG
    :members:
    :show-inheritance:


RSNAMICCAI
----------

.. currentmodule:: torchio.datasets.rsna_miccai

:class:`RSNAMICCAI`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: RSNAMICCAI
    :members:
    :show-inheritance:


MNI
---

.. automodule:: torchio.datasets.mni
.. currentmodule:: torchio.datasets.mni


:class:`ICBM2009CNonlinearSymmetric`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ICBM2009CNonlinearSymmetric
    :members:
    :show-inheritance:


:class:`Colin27`
~~~~~~~~~~~~~~~~

.. plot::

    import torchio as tio
    subject = tio.datasets.Colin27()
    subject.plot()

.. autoclass:: Colin27
    :members:
    :show-inheritance:


:class:`Pediatric`
~~~~~~~~~~~~~~~~~~

.. autoclass:: Pediatric
    :members:
    :show-inheritance:

.. plot::

    import torchio as tio
    subject = tio.datasets.Pediatric((4.5, 8.5))
    subject.plot()


:class:`Sheep`
~~~~~~~~~~~~~~

.. plot::

    import torchio as tio
    subject = tio.datasets.Sheep()
    subject.plot()

.. autoclass:: Sheep
    :members:
    :show-inheritance:


.. currentmodule:: torchio.datasets.bite

:class:`BITE3`
~~~~~~~~~~~~~~

.. autoclass:: BITE3
    :members:
    :show-inheritance:


ITK-SNAP
--------

.. automodule:: torchio.datasets.itk_snap
.. currentmodule:: torchio.datasets.itk_snap


:class:`BrainTumor`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: BrainTumor
    :members:
    :show-inheritance:


:class:`T1T2`
~~~~~~~~~~~~~

.. plot::

    import torchio as tio
    subject = tio.datasets.T1T2()
    subject.plot()

.. autoclass:: T1T2
    :members:
    :show-inheritance:


:class:`AorticValve`
~~~~~~~~~~~~~~~~~~~~

.. plot::

    import torchio as tio
    subject = tio.datasets.AorticValve()
    subject.plot()

.. autoclass:: AorticValve
    :members:
    :show-inheritance:


3D Slicer
---------

.. automodule:: torchio.datasets.slicer
.. currentmodule:: torchio.datasets.slicer

:class:`Slicer`
~~~~~~~~~~~~~~~

.. plot::

    import torchio as tio
    subject = tio.datasets.Slicer()
    subject.plot()

.. autoclass:: Slicer
    :members:
    :show-inheritance:


FPG
---

.. plot::

    import torchio as tio
    subject = tio.datasets.FPG()
    subject.plot()

.. currentmodule:: torchio.datasets.fpg

.. autoclass:: FPG
    :members:
    :show-inheritance:
