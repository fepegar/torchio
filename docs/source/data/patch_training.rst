Training
========

Random samplers
---------------

TorchIO includes grid, uniform and label patch samplers. There is also an
aggregator used for dense predictions.
For more information about patch-based training, see
`this NiftyNet tutorial <https://niftynet.readthedocs.io/en/dev/window_sizes.html>`_.


.. currentmodule:: torchio.data


:class:`ImageSampler`
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ImageSampler
    :members:
    :show-inheritance:


:class:`LabelSampler`
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: LabelSampler
    :members:
    :show-inheritance:


Queue
-----

In the following animation, :attr:`shuffle_subjects` is ``False``
and :attr:`shuffle_patches` is ``True``.

.. raw:: html

    <embed>
        <iframe style="width: 640px; height: 360px; overflow: hidden;"  scrolling="no" frameborder="0" src="https://editor.p5js.org/embed/DZwjZzkkV"></iframe>
    </embed>


.. currentmodule:: torchio.data

:class:`Queue`
^^^^^^^^^^^^^^

.. autoclass:: Queue
    :members:
    :show-inheritance:
