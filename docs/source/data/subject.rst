#######
Subject
#######

.. currentmodule:: torchio

The :class:`Subject` is a data structure used to store
images associated with a subject and any other metadata necessary for
processing.

All transforms applied to a :class:`Subject` are saved
in its :attr:`history` attribute (see :ref:`Reproducibility`).

.. autoclass:: Subject
    :members:
    :show-inheritance:
