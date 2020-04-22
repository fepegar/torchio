Training
========

Random samplers
---------------

TorchIO includes grid, uniform and label patch samplers. There is also an
aggregator used for dense predictions.
For more information about patch-based training, see
`this NiftyNet tutorial <https://niftynet.readthedocs.io/en/dev/window_sizes.html>`_.

Here's an example that uses a grid sampler and aggregator to perform dense
inference across a 3D volume using image patches::

    >>> import torch
    >>> import torch.nn as nn
    >>> import torchio
    >>> CHANNELS_DIMENSION = 1
    >>> patch_overlap = 4
    >>> patch_size = 128
    >>> grid_sampler = torchio.inference.GridSampler(
    ...     input_data,  # some PyTorch tensor or NumPy array
    ...     patch_size,
    ...     patch_overlap,
    ... )
    >>> patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=4)
    >>> aggregator = torchio.inference.GridAggregator(
    ...     input_data,  # some PyTorch tensor or NumPy array
    ...     patch_overlap,
    ... )
    >>> model = nn.Module()
    >>> model.to(device)
    >>> model.eval()
    >>> with torch.no_grad():
    ...     for patches_batch in patch_loader:
    ...         input_tensor = patches_batch[torchio.IMAGE].to(device)
    ...         locations = patches_batch[torchio.LOCATION]
    ...         logits = model(input_tensor)
    ...         labels = logits.argmax(dim=CHANNELS_DIMENSION, keepdim=True)
    ...         outputs = labels
    ...         aggregator.add_batch(outputs, locations)
    >>> output_tensor = aggregator.get_output_tensor()


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
