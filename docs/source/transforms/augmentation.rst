Augmentation
============

Augmentation transforms generate different results every time they are called.
The result can be made deterministic using the :py:attr:`seed` parameter.

.. contents::
    :local:


Base class
----------


.. currentmodule:: torchio.transforms.augmentation

:class:`RandomTransform`
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: RandomTransform
    :members:
    :show-inheritance:


.. currentmodule:: torchio.transforms


Composition
-----------


:class:`Compose`
^^^^^^^^^^^^^^^^

.. autoclass:: Compose
    :show-inheritance:


:class:`OneOf`
^^^^^^^^^^^^^^

.. autoclass:: OneOf
    :show-inheritance:



Spatial
-------

:class:`RandomFlip`
^^^^^^^^^^^^^^^^^^^

.. autoclass:: RandomFlip
    :show-inheritance:


:class:`RandomAffine`
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: RandomAffine
    :show-inheritance:


:class:`RandomElasticDeformation`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../../images/random_elastic_deformation.gif
    :alt: Random elastic deformation

.. autoclass:: RandomElasticDeformation
    :show-inheritance:


:class:`RandomDownsample`
^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: https://user-images.githubusercontent.com/12688084/87075276-fe9d9d00-c217-11ea-81a4-db0cac163ce7.png
    :alt: Simulation of an image with highly anisotropic spacing

.. autoclass:: RandomDownsample
    :show-inheritance:


Intensity
---------

:class:`RandomMotion`
^^^^^^^^^^^^^^^^^^^^^

.. image:: ../../images/random_motion.gif
   :alt: MRI k-space motion artifacts

.. autoclass:: RandomMotion
    :show-inheritance:


:class:`RandomGhosting`
^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../../images/random_ghosting.gif
   :alt: MRI k-space ghosting artifacts

.. autoclass:: RandomGhosting
    :show-inheritance:


:class:`RandomSpike`
^^^^^^^^^^^^^^^^^^^^

.. image:: ../../images/random_spike.gif
   :alt: MRI k-space spike artifacts

.. autoclass:: RandomSpike
    :show-inheritance:


:class:`RandomBiasField`
^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../../images/random_bias_field.gif
   :alt: MRI bias field artifact

.. autoclass:: RandomBiasField
    :show-inheritance:


:class:`RandomBlur`
^^^^^^^^^^^^^^^^^^^

.. autoclass:: RandomBlur
    :show-inheritance:


:class:`RandomNoise`
^^^^^^^^^^^^^^^^^^^^

.. image:: ../../images/random_noise.gif
   :alt: Random Gaussian noise

.. autoclass:: RandomNoise
    :show-inheritance:


:class:`RandomSwap`
^^^^^^^^^^^^^^^^^^^

.. image:: ../../images/random_swap.jpg
   :alt: Random patches swapping

.. autoclass:: RandomSwap
    :show-inheritance:


:class:`RandomLabelsToImage`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: RandomLabelsToImage
    :show-inheritance:
