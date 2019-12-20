import time
import multiprocessing as mp

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from torchio import ImagesDataset, Queue
from torchio.sampler import ImageSampler
from torchio.utils import create_dummy_dataset
from torchio.transforms import (
    ZNormalization,
    RandomNoise,
    RandomFlip,
    RandomAffine,
)


def main():
    # Define training and patches sampling parameters
    num_epochs = 4
    patch_size = 128
    queue_length = 100
    samples_per_volume = 10
    batch_size = 4

    def model(batch, sleep_time=0.1):
        """Dummy function to simulate a forward pass through the network"""
        time.sleep(sleep_time)
        return

    # Create a dummy dataset in the temporary directory, for this example
    paths_dict = create_dummy_dataset(
        num_images=100,
        size_range=(193, 229),
        force=False,
    )
    # paths_dict looks like this:
    # paths_dict = {'image': images_paths_list, 'label': labels_paths_list}

    # Define transforms for data normalization and augmentation
    transforms = (
        ZNormalization(),
        RandomNoise(std_range=(0, 0.25)),
        RandomAffine(scales=(0.9, 1.1), angles=(-10, 10)),
        RandomFlip(axes=(0,)),
    )
    transform = Compose(transforms)
    subjects_dataset = ImagesDataset(paths_dict, transform=transform)

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
            shuffle_patches=False,
            verbose=True
        )
        batch_loader = DataLoader(queue_dataset, batch_size=batch_size, shuffle=True)

        start = time.time()
        for epoch_index in range(num_epochs):
            print('Epoch {}'.format(epoch_index))
            for batch in batch_loader:
                logits = model(batch)
        print('Time:', int(time.time() - start), 'seconds')
        print()


if __name__ == "__main__":
    main()
