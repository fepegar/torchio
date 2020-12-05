#######
Subject
#######

The :class:`~torchio.data.Subject` is a data structure used to store
images associated with a subject and any other metadata necessary for
processing.

All transforms applied to a :class:`~torchio.data.Subject` are saved
in its :attr:`history` attribute (see :ref:`Reproducibility`).

.. currentmodule:: torchio.data

.. autoclass:: Subject
    :members:
    :show-inheritance:
