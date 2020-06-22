Inference
=========


Here's an example that uses a grid sampler and aggregator to perform dense
inference across a 3D image using small patches::

    >>> import torch
    >>> import torch.nn as nn
    >>> import torchio
    >>> patch_overlap = 4, 4, 4  # or just 4
    >>> patch_size = 88, 88, 60
    >>> sample = torchio.datasets.Colin27()
    >>> sample
    Colin27(Keys: ('t1', 'head', 'brain'); images: 3)
    >>> grid_sampler = torchio.inference.GridSampler(
    ...     sample,
    ...     patch_size,
    ...     patch_overlap,
    ... )
    >>> patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=4)
    >>> aggregator = torchio.inference.GridAggregator(grid_sampler)
    >>> model = nn.Identity()
    >>> model.eval()
    >>> with torch.no_grad():
    ...     for patches_batch in patch_loader:
    ...         input_tensor = patches_batch['t1'][torchio.DATA].to(device)
    ...         locations = patches_batch[torchio.LOCATION]
    ...         logits = model(input_tensor)
    ...         labels = logits.argmax(dim=torchio.CHANNELS_DIMENSION, keepdim=True)
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
