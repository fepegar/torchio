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

:class:`Rescale`
~~~~~~~~~~~~~~~~

.. autoclass:: Rescale
    :show-inheritance:


:class:`ZNormalization`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ZNormalization
    :show-inheritance:


:class:`HistogramStandardization`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: HistogramStandardization
    :show-inheritance:


Spatial
-------

:class:`CenterCropOrPad`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CenterCropOrPad
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
