*******
Example
*******

Here's an example showcasing multiple features in Torchio.

For a more complete example including training of a 3D U-Net
for brain segmentation, see the :ref:`Google Colab Jupyter Notebook <colab_notebok>`.

.. code-block:: python

    import time
    import multiprocessing as mp

    from tqdm import trange

    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose

    from torchio import ImagesDataset, Queue, DATA
    from torchio.data.sampler import ImageSampler
    from torchio.utils import create_dummy_dataset
    from torchio.transforms import (
        ZNormalization,
        RandomNoise,
        RandomFlip,
        RandomAffine,
    )

    # Define training and patches sampling parameters
    num_epochs = 4
    patch_size = 128
    queue_length = 400
    samples_per_volume = 10
    batch_size = 4

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(
                in_channels=1,
                out_channels=3,
                kernel_size=3,
            )
        def forward(self, x):
            return self.conv(x)

    model = Network()

    # Create a dummy dataset in the temporary directory, for this example
    subjects_list = create_dummy_dataset(
        num_images=100,
        size_range=(193, 229),
        force=False,
    )

    # Each element of subjects_list is an instance of torchio.Subject:
    # subject = Subject(
    #     torchio.Image('one_image', path_to_one_image, torchio.INTENSITY),
    #     torchio.Image('another_image', path_to_another_image, torchio.INTENSITY),
    #     torchio.Image('a_label', path_to_a_label, torchio.LABEL),
    # )

    # Define transforms for data normalization and augmentation
    transforms = (
        ZNormalization(),
        RandomNoise(std_range=(0, 0.25)),
        RandomAffine(scales=(0.9, 1.1), degrees=10),
        RandomFlip(axes=(0,)),
    )
    transform = Compose(transforms)
    subjects_dataset = ImagesDataset(subjects_list, transform)

    # Run a benchmark for different numbers of workers
    workers = range(mp.cpu_count() + 1)
    for num_workers in workers:
        print('Number of workers:', num_workers)

        # Define the dataset as a queue of patches
        queue_dataset = Queue(
            subjects_dataset,
            queue_length,
            samples_per_volume,
            patch_size,
            ImageSampler,
            num_workers=num_workers,
        )
        batch_loader = DataLoader(queue_dataset, batch_size=batch_size)

        start = time.time()
        for epoch_index in trange(num_epochs, leave=False):
            for batch in batch_loader:
                # The keys of batch have been defined in create_dummy_dataset()
                inputs = batch['one_modality'][DATA]
                targets = batch['segmentation'][DATA]
                logits = model(inputs)
        print('Time:', int(time.time() - start), 'seconds')
        print()


Output::
    Number of workers: 0
    Time: 394 seconds

    Number of workers: 1
    Time: 372 seconds

    Number of workers: 2
    Time: 278 seconds

    Number of workers: 3
    Time: 259 seconds

    Number of workers: 4
    Time: 242 seconds
