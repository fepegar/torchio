import time
import multiprocessing as mp

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

import torchio
from torchio import ImagesDataset, Queue
from torchio.sampler import ImageSampler
from torchio.transforms import (
    ZNormalization,
    RandomNoise,
    RandomFlip,
    RandomAffine,
)

# Mock PyTorch model
model = lambda x: x


# Define training and patches sampling parameters
num_epochs = 4
patch_size = 128
queue_length = 100
samples_per_volume = 1
batch_size = 2

# Define transforms for data normalization and augmentation
transforms = (
    ZNormalization(),
    RandomAffine(scales=(0.9, 1.1), degrees=10),
    RandomNoise(std_range=(0, 0.25)),
    RandomFlip(axes=(0,)),
)
transform = Compose(transforms)

# Populate a list with dictionaries of paths
one_subject_dict = {
    'T1': dict(path='../BRATS2018_crop_renamed/LGG75_T1.nii.gz', type=torchio.INTENSITY),
    'T2': dict(path='../BRATS2018_crop_renamed/LGG75_T2.nii.gz', type=torchio.INTENSITY),
    'label': dict(path='../BRATS2018_crop_renamed/LGG75_Label.nii.gz', type=torchio.INTENSITY),
}

another_subject_dict = {
    'T1': dict(path='../BRATS2018_crop_renamed/LGG74_T1.nii.gz', type=torchio.INTENSITY),
    'label': dict(path='../BRATS2018_crop_renamed/LGG74_Label.nii.gz', type=torchio.INTENSITY),
}

subjects_paths = [
    one_subject_dict,
    another_subject_dict,
]

subjects_dataset = ImagesDataset(subjects_paths, transform=transform)

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
        shuffle_subjects=False,
    )

    # This collate_fn is needed in the case of missing modalities (TODO: elaborate)
    batch_loader = DataLoader(
        queue_dataset, batch_size=batch_size, collate_fn=lambda x: x)

    start = time.time()
    for epoch_index in range(num_epochs):
        for batch in batch_loader:
            logits = model(batch)
            print([batch[idx].keys() for idx in range(batch_size)])
    print('Time:', int(time.time() - start), 'seconds')
    print()
