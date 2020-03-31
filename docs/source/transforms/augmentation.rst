Augmentation
============

Augmentation transforms generate different results every time they are called.
The result can be made deterministic using the :py:attr:`seed` parameter.

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

.. image:: ../../images/random_elastic_deformation.gif
   :alt: Random elastic deformation

.. autoclass:: RandomElasticDeformation
    :show-inheritance:



Intensity
---------

:class:`RandomMotion`
~~~~~~~~~~~~~~~~~~~~~~~~~

Magnetic resonance images suffer from motion artifacts when the subject moves
during image acquisition. This transform follows
`Shaw et al., 2019 <http://proceedings.mlr.press/v102/shaw19a.html>`_ to
simulate motion artifacts for data augmentation.

.. image:: ../../images/random_motion.gif
   :alt: MRI k-space motion artifacts

.. autoclass:: RandomMotion
    :show-inheritance:


:class:`RandomGhosting`
~~~~~~~~~~~~~~~~~~~~~~~

Discrete "ghost" artifacts may occur along the phase-encode direction whenever
the position or signal intensity of imaged structures within the field-of-view
vary or move in a regular (periodic) fashion.
Pulsatile flow of blood or CSF, cardiac motion, and respiratory motion are the
most important patient-related causes of ghost artifacts in clinical MR imaging
(from `mriquestions.com <http://mriquestions.com/why-discrete-ghosts.html>`_).

.. image:: ../../images/random_ghosting.gif
   :alt: MRI k-space ghosting artifacts

.. autoclass:: RandomGhosting
    :show-inheritance:


:class:`RandomSpike`
~~~~~~~~~~~~~~~~~~~~

Also known as
`Herringbone artifact <https://radiopaedia.org/articles/herringbone-artifact?lang=gb>`_,
crisscross artifact or corduroy artifact, it creates stripes in different
directions in image space due to spikes in k-space.

.. image:: ../../images/random_spike.gif
   :alt: MRI k-space spike artifacts

.. autoclass:: RandomSpike
    :show-inheritance:


:class:`RandomBiasField`
~~~~~~~~~~~~~~~~~~~~~~~~

MRI magnetic field inhomogeneity creates intensity
variations of very low frequency across the whole image.

The bias field is modelled as a linear combination of
polynomial basis functions, as in K. Van Leemput et al., 1999,
*Automated model-based tissue classification of MR images of the brain*.

It was implemented in NiftyNet by Carole Sudre and used in
`Sudre et al., 2017, Longitudinal segmentation of age-related white matter
hyperintensities <https://www.sciencedirect.com/science/article/pii/S1361841517300257?via%3Dihub>`_.

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

Adds noise sampled from a normal distribution with mean 0 and standard
deviation sampled from a uniform distribution in the range `std_range`.
It is often used after :py:class:`~torchio.transforms.ZNormalization`,
because its output has zero-mean.

.. image:: ../../images/random_noise.gif
   :alt: Random Gaussian noise

.. autoclass:: RandomNoise
    :show-inheritance:


:class:`RandomSwap`
~~~~~~~~~~~~~~~~~~~

Randomly swaps patches in the image.
This is typically used in
`context restoration for self-supervised learning
<https://www.sciencedirect.com/science/article/pii/S1361841518304699>`_.

.. image:: ../../images/random_swap.jpg
   :alt: Random patches swapping

.. autoclass:: RandomSwap
    :show-inheritance:
