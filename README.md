# torchio

## Installation

I recommend cloning and doing an editable installation, as this is still very
experimental and changes very often.

```
git clone https://github.com/fepegar/torchio.git
pip install --editable torchio
```

## Examples

### [Training](examples/example_queue.py)

```python
import time
import multiprocessing as mp

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from torchio.sampler import ImageSampler
from torchio.utils import create_dummy_dataset
from torchio import ImagesDataset, Queue
from torchio.transforms import (
    ZNormalization,
    RandomNoise,
    RandomFlip,
    RandomAffine,
)


# Define training and patches sampling parameters
num_epochs = 4
patch_size = 128
queue_length = 100
samples_per_volume = 10
batch_size = 4

def model(batch, sleep_time=0.1):
    """Dummy function to simulate a forward pass through the network"""
    time.sleep(sleep_time)
    return batch

# Create a dummy dataset in the temporary directory, for this example
subjects_paths = create_dummy_dataset(
    num_images=100,
    size_range=(193, 229),
    force=False,
)

# Each element of subjects_paths is a dictionary:
# subject = {
#     'one_image': dict(path=path_to_one_image, type=torchio.INTENSITY),
#     'another_image': dict(path=path_to_another_image, type=torchio.INTENSITY),
#     'a_label': dict(path=path_to_a_label, type=torchio.LABEL),
# )

# Define transforms for data normalization and augmentation
transforms = (
    ZNormalization(),
    RandomNoise(std_range=(0, 0.25)),
    RandomAffine(scales=(0.9, 1.1), angles=(-10, 10)),
    RandomFlip(axes=(0,)),
)
transform = Compose(transforms)
subjects_dataset = ImagesDataset(subjects_paths, transform)

sample = subjects_dataset[0]

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
    for epoch_index in range(num_epochs):
        for batch in batch_loader:
            logits = model(batch)
    print('Time:', int(time.time() - start), 'seconds')
    print()
```


Output:
```
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
```


### [Inference](examples/example_inference.py)

```python
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchio.inference import GridSampler, GridAggregator

def model(arg):
    """Mock PyTorch model"""
    return arg

patch_size = 128, 128, 128
patch_overlap = 4, 4, 4
batch_size = 6
CHANNELS_DIMENSION = 1

# Let's create a dummy volume
input_array = torch.rand((193, 229, 193)).numpy()

# More info about patch-based inference in NiftyNet docs:
# https://niftynet.readthedocs.io/en/dev/window_sizes.html
grid_sampler = GridSampler(input_array, patch_size, patch_overlap)
aggregator = GridAggregator(input_array, patch_overlap)
patch_loader = DataLoader(grid_sampler, batch_size=batch_size)

with torch.no_grad():
    for patches_batch in tqdm(patch_loader):
        input_tensor = patches_batch['image']
        locations = patches_batch['location']
        logits = model(input_tensor)  # some model
        labels = logits.argmax(dim=CHANNELS_DIMENSION, keepdim=True)
        outputs = labels
        aggregator.add_batch(outputs, locations)

output_array = aggregator.output_array
```
