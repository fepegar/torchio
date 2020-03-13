Augmentation
============

.. contents::
   :local:

.. currentmodule:: torchio.transforms.augmentation

:class:`RandomTransform`
------------------------

.. autoclass:: RandomTransform
    :members:
    :show-inheritance:


.. currentmodule:: torchio.transforms



Spatial
-------

:class:`RandomFlip`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomFlip
    :show-inheritance:


:class:`RandomAffine`
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomAffine
    :show-inheritance:


:class:`RandomElasticDeformation`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomElasticDeformation
    :show-inheritance:



Intensity
---------

:class:`RandomMotion`
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomMotion
    :show-inheritance:


:class:`RandomGhosting`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomGhosting
    :show-inheritance:


:class:`RandomSpike`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomSpike
    :show-inheritance:


:class:`RandomBiasField`
~~~~~~~~~~~~~~~~~~~~~~~~

MRI magnetic field inhomogeneity creates intensity
variations of very low frequency across the whole image.

The bias field is modelled as a linear combination of
polynomial basis functions, as in K. Van Leemput et al., 1999,
*Automated model-based tissue classification of MR images of the brain*.

It was added to NiftyNet by Carole Sudre and used in
C. Sudre et al., 2017, *Longitudinal segmentation of age-related
white matter hyperintensities*.

.. image:: ../../images/random_bias_field.gif
   :alt: MRI bias field artifact

.. autoclass:: RandomBiasField
    :show-inheritance:


:class:`RandomBlur`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomBlur
    :show-inheritance:


:class:`RandomNoise`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomNoise
    :show-inheritance:


:class:`RandomSwap`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomSwap
    :show-inheritance:
