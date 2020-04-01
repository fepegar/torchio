Preprocessing
=============

.. contents::
   :local:

Intensity
---------

.. currentmodule:: torchio.transforms.preprocessing.intensity

:class:`NormalizationTransform`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NormalizationTransform
    :show-inheritance:


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

Implementation of
`New variants of a method of MRI scale standardization <https://ieeexplore.ieee.org/document/836373>`_,
adapted from NiftyNet.

.. image:: ../../images/histogram_standardization.png
   :alt: Histogram standardization

.. autoclass:: HistogramStandardization
    :show-inheritance:
    :members:


Spatial
-------

:class:`CropOrPad`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CropOrPad
    :show-inheritance:
    :members: _get_six_bounds_parameters


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
