# [TorchIO](http://torchio.rtfd.io/)

[![PyPI downloads](https://img.shields.io/pypi/dm/torchio.svg?label=PyPI%20downloads&logo=python&logoColor=white)](https://pypi.org/project/torchio/)
[![PyPI version](https://badge.fury.io/py/torchio.svg)](https://badge.fury.io/py/torchio)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/112NTL8uJXzcMw4PQbUvMQN-WHlVwQS3i)
[![Build status](https://img.shields.io/travis/fepegar/torchio/master.svg?label=Travis%20CI%20build&logo=travis)](https://travis-ci.org/fepegar/torchio)
[![Coverage status](https://codecov.io/gh/fepegar/torchio/branch/master/graphs/badge.svg)](https://codecov.io/github/fepegar/torchio)
[![Code quality](https://img.shields.io/scrutinizer/g/fepegar/torchio.svg?label=Code%20quality&logo=scrutinizer)](https://scrutinizer-ci.com/g/fepegar/torchio/?branch=master)
[![Code maintainability](https://api.codeclimate.com/v1/badges/518673e49a472dd5714d/maintainability)](https://codeclimate.com/github/fepegar/torchio/maintainability)
[![Slack](https://img.shields.io/badge/TorchIO-Join%20on%20Slack-blueviolet?style=flat&logo=slack)](https://join.slack.com/t/torchioworkspace/shared_invite/enQtOTY1NTgwNDI4NzA1LTEzMjIwZTczMGRmM2ZlMzBkZDg3YmQwY2E4OTIyYjFhZDVkZmIwOWZkNTQzYTFmYzdiNGEwZWQ4YjgwMTczZmE)

---

### 🎉 News: the paper is out! 🎉

See the [Credits](#credits) section for more information.

---

`torchio` is a Python package containing a set of tools to efficiently
read, sample and write 3D medical images in deep learning applications
written in [PyTorch](https://pytorch.org/),
including intensity and spatial transforms
for data augmentation and preprocessing. Transforms include typical computer vision operations
such as random affine transformations and also domain-specific ones such as
simulation of intensity artifacts due to
[MRI magnetic field inhomogeneity](http://mriquestions.com/why-homogeneity.html)
or [k-space motion artifacts](http://proceedings.mlr.press/v102/shaw19a.html).

This package has been greatly inspired by [NiftyNet](https://niftynet.io/).

## Jupyter notebook

See the [docs](https://torchio.readthedocs.io/quickstart.html#google-colab-jupyter-notebok).


## Credits

If you like this repository, please click on Star!

If you use this package for your research, please cite the paper:

[Pérez-García et al., 2020, *TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning*](https://arxiv.org/abs/2003.04696).

BibTeX entry:

```bibtex
@misc{fern2020torchio,
    title={TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning},
    author={Fernando Pérez-García and Rachel Sparks and Sebastien Ourselin},
    year={2020},
    eprint={2003.04696},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```


## Installation

This package is on the
[Python Package Index (PyPI)](https://pypi.org/project/torchio/).
To install the latest published version, just run the following command in a terminal:

```shell
$ pip install --upgrade torchio
```

## [Documentation](https://torchio.readthedocs.io/)

The documentation is a work in progress, but some classes such as
[`ImagesDataset`](https://torchio.readthedocs.io/data/image.html)
are already fairly well documented.

The examples in this README will soon be moved to the docs.


## Index

- [Features](#features)
  * [Medical image datasets](#medical-image-datasets)
    - [IXI](#ixi)
    - [Tiny IXI](#tiny-ixi)
  * [Data handling](#data-handling)
    - [`ImagesDataset`](#imagesdataset)
    - [Samplers and aggregators](#samplers-and-aggregators)
    - [`Queue`](#queue)
  * [Transforms](#transforms)
    - [Augmentation](#augmentation)
      * [Intensity](#intensity)
        - [MRI k-space motion artifacts](#mri-k-space-motion-artifacts)
        - [MRI k-space ghosting artifacts](#mri-k-space-ghosting-artifacts)
        - [MRI k-space spike artifacts](#mri-k-space-spike-artifacts)
        - [MRI magnetic field inhomogeneity](#mri-magnetic-field-inhomogeneity)
        - [Patch swap](#patch-swap)
        - [Gaussian noise](#gaussian-noise)
        - [Gaussian blurring](#gaussian-blurring)
      * [Spatial](#spatial)
        - [B-spline dense elastic deformation](#b-spline-dense-elastic-deformation)
        - [Flip](#flip)
        - [Affine transform](#affine-transform)
    - [Preprocessing](#preprocessing)
      * [Histogram standardization](#histogram-standardization)
      * [Z-normalization](#z-normalization)
      * [Rescale](#rescale)
      * [Resample](#resample)
      * [Pad](#pad)
      * [Crop](#crop)
      * [ToCanonical](#tocanonical)
      * [CenterCropOrPad](#centercroporpad)
    - [Others](#others)
      * [Lambda](#lambda)


- [Example](#example)
- [Related projects](#related-projects)
- [See also](#see-also)





## Features


### Medical image datasets

#### [IXI](https://torchio.readthedocs.io/datasets.html#id1)

#### [Tiny IXI](https://torchio.readthedocs.io/datasets.html#ixitiny)

### Data handling

#### [`ImagesDataset`](https://torchio.readthedocs.io/data/images.html#imagesdataset)


#### [Samplers and aggregators](torchio/data/sampler/sampler.py)

TorchIO includes grid, uniform and label patch samplers. There is also an
aggregator used for dense predictions.
For more information about patch-based training, see
[NiftyNet docs](https://niftynet.readthedocs.io/en/dev/window_sizes.html).

```python
import torch
import torch.nn as nn
import torchio

CHANNELS_DIMENSION = 1
patch_overlap = 4
patch_size = 128

grid_sampler = torchio.inference.GridSampler(
    input_data,  # some PyTorch tensor or NumPy array
    patch_size,
    patch_overlap,
)
patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=4)
aggregator = torchio.inference.GridAggregator(
    input_data,  # some PyTorch tensor or NumPy array
    patch_overlap,
)

model = nn.Module()
model.to(device)
model.eval()
with torch.no_grad():
    for patches_batch in patch_loader:
        input_tensor = patches_batch['image'].to(device)
        locations = patches_batch['location']
        logits = model(input_tensor)
        labels = logits.argmax(dim=CHANNELS_DIMENSION, keepdim=True)
        outputs = labels
        aggregator.add_batch(outputs, locations)

output_tensor = aggregator.get_output_tensor()
```


#### [`Queue`](torchio/data/queue.py)

A patches `Queue` (or buffer) can be used for randomized patch-based sampling
during training.
[This interactive animation](https://niftynet.readthedocs.io/en/dev/config_spec.html#queue-length)
can be used to understand how the queue works.

```python
import torch
import torchio

patches_queue = torchio.Queue(
    subjects_dataset=subjects_dataset,  # instance of torchio.ImagesDataset
    max_length=300,
    samples_per_volume=10,
    patch_size=96,
    sampler_class=torchio.sampler.ImageSampler,
    num_workers=4,
    shuffle_subjects=True,
    shuffle_patches=True,
)
patches_loader = DataLoader(patches_queue, batch_size=4)

num_epochs = 20
for epoch_index in range(num_epochs):
    for patches_batch in patches_loader:
        logits = model(patches_batch)  # model is some torch.nn.Module
```


### [Transforms](https://torchio.readthedocs.io/transforms/transforms.html)

#### Augmentation

##### Intensity

###### [MRI k-space motion artifacts](https://torchio.readthedocs.io/transforms/augmentation.html#torchio.transforms.RandomMotion)

![MRI k-space motion artifacts](https://raw.githubusercontent.com/fepegar/torchio/master/docs/images/random_motion.gif)


###### [MRI k-space ghosting artifacts](https://torchio.readthedocs.io/transforms/augmentation.html#randomghosting)

![MRI k-space ghosting artifacts](https://raw.githubusercontent.com/fepegar/torchio/master/docs/images/random_ghosting.gif)


###### [MRI k-space spike artifacts](https://torchio.readthedocs.io/transforms/augmentation.html#randomspike)

![MRI k-space spike artifacts](https://raw.githubusercontent.com/fepegar/torchio/master/docs/images/random_spike.gif)


###### [MRI magnetic field inhomogeneity](https://torchio.readthedocs.io/transforms/augmentation.html#randombiasfield)

![MRI bias field artifact](https://raw.githubusercontent.com/fepegar/torchio/master/docs/images/random_bias_field.gif)


##### [Patch swap](https://torchio.readthedocs.io/transforms/augmentation.html#randomswap)

![Random patches swapping](https://raw.githubusercontent.com/fepegar/torchio/master/docs/images/random_swap.jpg)


###### [Gaussian noise](https://torchio.readthedocs.io/transforms/augmentation.html#randomnoise)

![Random Gaussian noise](https://raw.githubusercontent.com/fepegar/torchio/master/docs/images/random_noise.gif)


###### [Gaussian blurring](https://torchio.readthedocs.io/transforms/augmentation.html#randomblur)


##### Spatial

###### [B-spline dense elastic deformation](https://torchio.readthedocs.io/transforms/augmentation.html#randomelasticdeformation)

<p align="center">
  <img src="https://raw.githubusercontent.com/fepegar/torchio/master/docs/images/random_elastic_deformation.gif" alt="Random elastic deformation"/>
</p>

###### [Flip](thttps://torchio.readthedocs.io/transforms/augmentation.html#randomflip)

###### [Affine transform](https://torchio.readthedocs.io/transforms/augmentation.html#randomaffine)


#### Preprocessing

##### [Histogram standardization](https://torchio.readthedocs.io/transforms/preprocessing.html#histogramstandardization)

##### [Rescale](https://torchio.readthedocs.io/transforms/preprocessing.html#rescale)

##### [Z-normalization](https://torchio.readthedocs.io/transforms/preprocessing.html#znormalization)

##### [Resample](https://torchio.readthedocs.io/transforms/preprocessing.html#resample)

##### [Pad](https://torchio.readthedocs.io/transforms/preprocessing.html#pad)

##### [Crop](https://torchio.readthedocs.io/transforms/preprocessing.html#crop)

##### [ToCanonical](https://torchio.readthedocs.io/transforms/preprocessing.html#tocanonical)

##### [CenterCropOrPad](https://torchio.readthedocs.io/transforms/preprocessing.html#centercroporpad)


#### Others

##### [Lambda](https://torchio.readthedocs.io/transforms/others.html#lambda)



## [Example](examples/example_queue.py)

This example shows the improvement in performance when multiple workers are
used to load and preprocess the volumes using multiple workers.

```python
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
```


Output:
```python
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


## Related projects

* [Albumentations](https://github.com/albumentations-team/albumentations)
* [`batchgenerators`](https://github.com/MIC-DKFZ/batchgenerators)
* [kornia](https://kornia.github.io/)
* [DALI](https://developer.nvidia.com/DALI)
* [`rising`](https://github.com/PhoenixDL/rising)


## See also

* [`highresnet`](https://www.github.com/fepegar/highresnet)
* [`unet`](https://www.github.com/fepegar/unet)
