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

.. plot::

    import torchio as tio
    subject = tio.datasets.BrainTumor()
    subject.plot()

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
