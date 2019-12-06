# torchio

## Installation

I recommend cloning and doing an editable installation, as this is still very
experimental and changes very often.

```
git clone https://github.com/fepegar/torchio.git
pip install --editable torchio
```

## Example

```python
import time
import shutil
import tempfile
from pathlib import Path
import multiprocessing as mp

import numpy as np
import nibabel as nib
from tqdm import tqdm, trange

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from torchio import ImagesDataset, Queue
from torchio.sampler import ImageSampler
from torchio.transforms import RandomFlip, RandomAffine


def create_dummy_dataset(num_images, size_range, force=False):
    tempdir = Path(tempfile.gettempdir())
    images_dir = tempdir / 'dummy_images'
    labels_dir = tempdir / 'dummy_labels'

    if force:
        shutil.rmtree(images_dir)
        shutil.rmtree(labels_dir)

    if not images_dir.is_dir():
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        for i in trange(num_images):
            shape = np.random.randint(*size_range, size=3)
            affine = np.eye(4)
            image = np.random.rand(*shape)
            label = np.ones_like(image)
            label[image < 0.33] = 0
            label[image > 0.66] = 2
            image *= 255
            nii = nib.Nifti1Image(image.astype(np.uint8), affine)
            nii.to_filename(str(images_dir / f'image_{i}.nii.gz'))
            nii = nib.Nifti1Image(label.astype(np.uint8), affine)
            nii.to_filename(str(labels_dir / f'label_{i}.nii.gz'))
    image_paths = sorted(list(images_dir.glob('*.nii*')))
    label_paths = sorted(list(labels_dir.glob('*.nii*')))
    paths_dict = dict(image=image_paths, label=label_paths)
    return paths_dict

def main():
    # Config
    force = False

    num_images = 100
    size_range = 193, 229
    paths_dict = create_dummy_dataset(num_images, size_range, force=force)

    verbose = False
    patch_size = 128

    queue_length = 100
    samples_per_volume = 10
    batch_size = 4

    # Run the benchmark
    scales = (0.9, 1.1)
    angles = (-10, 10)
    axes = (0,)

    transforms = (
        RandomAffine(scales=scales, angles=angles, isotropic=False, verbose=False),
        RandomFlip(axes, verbose=False),
    )
    transform = Compose(transforms)
    subjects_dataset = ImagesDataset(
        paths_dict, transform=transform, verbose=verbose)

    for num_workers in range(mp.cpu_count() + 1):
        print('Number of workers:', num_workers)
        queue = Queue(
            subjects_dataset,
            queue_length,
            samples_per_volume,
            patch_size,
            ImageSampler,
            num_workers=num_workers,
            shuffle_dataset=False,
            verbose=verbose,
        )

        loader = DataLoader(queue, batch_size=batch_size)

        start = time.time()
        progress = tqdm(loader) if verbose else loader
        for batch in progress:
            time.sleep(0.25)  # simulate forward and backward pass
        print('Time:', int(time.time() - start), 'seconds')
        print()


if __name__ == "__main__":
    main()
```


Output:
```
Number of workers: 0
Time: 185 seconds

Number of workers: 1
Time: 192 seconds

Number of workers: 2
Time: 147 seconds

Number of workers: 3
Time: 153 seconds

Number of workers: 4
Time: 130 seconds
```
