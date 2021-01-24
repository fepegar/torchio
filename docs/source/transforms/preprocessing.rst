Preprocessing
=============

Intensity
---------

.. currentmodule:: torchio.transforms


:class:`RescaleIntensity`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RescaleIntensity
    :show-inheritance:


:class:`ZNormalization`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ZNormalization
    :show-inheritance:


:class:`HistogramStandardization`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../images/histogram_standardization.png
   :alt: Histogram standardization

.. autoclass:: HistogramStandardization
    :show-inheritance:
    :members:


.. currentmodule:: torchio.transforms.preprocessing.intensity


:class:`NormalizationTransform`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NormalizationTransform
    :show-inheritance:


Spatial
-------

.. currentmodule:: torchio.transforms

:class:`CropOrPad`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CropOrPad
    :show-inheritance:
    :members: _get_six_bounds_parameters


:class:`EnsureShapeMultiple`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: EnsureShapeMultiple
    :show-inheritance:


:class:`Crop`
~~~~~~~~~~~~~

.. autoclass:: Crop
    :show-inheritance:


:class:`Pad`
~~~~~~~~~~~~

.. autoclass:: Pad
    :show-inheritance:


:class:`Resample`
~~~~~~~~~~~~~~~~~

.. autoclass:: Resample
    :show-inheritance:


:class:`ToCanonical`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ToCanonical
    :show-inheritance:


Label
---------


:class:`RemapLabels`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RemapLabels
    :show-inheritance:


:class:`RemoveLabels`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RemoveLabels
    :show-inheritance:


:class:`SequentialLabels`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SequentialLabels
    :show-inheritance:


:class:`OneHot`
~~~~~~~~~~~~~~~

.. autoclass:: OneHot
    :show-inheritance:


:class:`Contour`
~~~~~~~~~~~~~~~~

.. autoclass:: Contour
    :show-inheritance:


:class:`KeepLargestComponent`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: KeepLargestComponent
    :show-inheritance:
