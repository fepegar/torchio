Inference
=========


Here's an example that uses a grid sampler and aggregator to perform dense
inference across a 3D image using small patches::

    >>> import torch
    >>> import torch.nn as nn
    >>> import torchio as tio
    >>> patch_overlap = 4, 4, 4  # or just 4
    >>> patch_size = 88, 88, 60
    >>> subject = tio.datasets.Colin27()
    >>> subject
    Colin27(Keys: ('t1', 'head', 'brain'); images: 3)
    >>> grid_sampler = tio.inference.GridSampler(
    ...     subject,
    ...     patch_size,
    ...     patch_overlap,
    ... )
    >>> patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=4)
    >>> aggregator = tio.inference.GridAggregator(grid_sampler)
    >>> model = nn.Identity().eval()
    >>> with torch.no_grad():
    ...     for patches_batch in patch_loader:
    ...         input_tensor = patches_batch['t1'][tio.DATA]
    ...         locations = patches_batch[tio.LOCATION]
    ...         logits = model(input_tensor)
    ...         labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
    ...         outputs = labels
    ...         aggregator.add_batch(outputs, locations)
    >>> output_tensor = aggregator.get_output_tensor()


Grid sampler
------------

.. currentmodule:: torchio.data

:class:`GridSampler`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GridSampler
    :members:
    :show-inheritance:


Grid aggregator
---------------

:class:`GridAggregator`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GridAggregator
    :members:
