# TorchIO

[![Downloads](https://img.shields.io/pypi/dm/torchio.svg)](https://pypi.org/project/torchio/)
[![PyPI version](https://badge.fury.io/py/torchio.svg)](https://badge.fury.io/py/torchio)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/112NTL8uJXzcMw4PQbUvMQN-WHlVwQS3i)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3598622.svg)](https://doi.org/10.5281/zenodo.3598622)
[![Build status](https://travis-ci.org/fepegar/torchio.svg?branch=master)](https://travis-ci.org/fepegar/torchio)
[![Documentation status](https://readthedocs.org/projects/torchio/badge/?version=latest)](https://torchio.readthedocs.io/?badge=latest)
[![Coverage status](https://codecov.io/gh/fepegar/torchio/branch/master/graphs/badge.svg)](https://codecov.io/github/fepegar/torchio)
[![Code quality](https://img.shields.io/scrutinizer/g/fepegar/torchio.svg)](https://scrutinizer-ci.com/g/fepegar/torchio/?branch=master)
[![Slack](https://img.shields.io/badge/TorchIO-Join%20on%20Slack-blueviolet?style=flat&logo=slack)](https://join.slack.com/t/torchioworkspace/shared_invite/enQtOTY1NTgwNDI4NzA1LTEzMjIwZTczMGRmM2ZlMzBkZDg3YmQwY2E4OTIyYjFhZDVkZmIwOWZkNTQzYTFmYzdiNGEwZWQ4YjgwMTczZmE)



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

The best way to quickly understand and try the library is the
[Jupyter notebook](https://colab.research.google.com/drive/112NTL8uJXzcMw4PQbUvMQN-WHlVwQS3i)
hosted by Google Colab.
It includes many examples and visualization of most of the classes and even
training of a [3D U-Net](https://www.github.com/fepegar/unet) for brain
segmentation of T1-weighted MRI with whole images and patch-based sampling.


## Credits

If you like this repository, please click on Star!

If you used this package for your research, please cite this repository using
the information available on its
[Zenodo entry](https://doi.org/10.5281/zenodo.3598622) or use this text:

> Pérez-García, Fernando.
(2020, January 15).
fepegar/torchio: TorchIO: Tools for loading, augmenting and writing 3D medical images on PyTorch. Zenodo.
http://doi.org/10.5281/zenodo.3598622

BibTeX entry:

```bibtex
@software{perez_garcia_fernando_2020_3598622,
  author       = {Pérez-García, Fernando},
  title        = {{fepegar/torchio: TorchIO: Tools for loading,
                   augmenting and writing 3D medical images on
                   PyTorch}},
  month        = jan,
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3598622},
  url          = {https://doi.org/10.5281/zenodo.3598622}
}
```

## Installation

This package is on the
[Python Package Index (PyPI)](https://pypi.org/project/torchio/).
To install the latest published version, just run the following command in a terminal:

```shell
$ pip install --upgrade torchio
```


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

#### [IXI](torchio/datasets/ixi.py)

The [Information eXtraction from Images (IXI)](https://brain-development.org/ixi-dataset/) dataset
contains "nearly 600 MR images from normal, healthy subjects", including
"T1, T2 and PD-weighted images, MRA images and Diffusion-weighted images (15 directions)".

The usage is very similar to [`torchvision.datasets`](https://pytorch.org/docs/stable/torchvision/datasets.html):
```python
import torchio
import torchvision

transforms = [
    torchio.ToCanonical(),  # to RAS
    torchio.Resample((1, 1, 1)),  # to 1 mm iso
]

ixi_dataset = torchio.datasets.IXI(
    'path/to/ixi_root/',
    modalities=('T1', 'T2'),
    transform=torchvision.transforms.Compose(transforms),
    download=True,
)
print('Number of subjects in dataset:', len(ixi_dataset))  # 577

sample_subject = ixi_dataset[0]
print('Keys in subject sample:', tuple(sample_subject.keys()))  # ('T1', 'T2')
print('Shape of T1 data:', sample_subject['T1'][torchio.DATA].shape)  # [1, 180, 268, 268]
print('Shape of T2 data:', sample_subject['T2'][torchio.DATA].shape)  # [1, 241, 257, 188]
```


#### [Tiny IXI](torchio/datasets/ixi.py)

This is the dataset used in the [notebook](https://colab.research.google.com/drive/112NTL8uJXzcMw4PQbUvMQN-WHlVwQS3i).
It is a tiny version of IXI, containing 566 T1-weighted brain MR images
and their corresponding brain segmentations, all with size (83 x 44 x 55).


### Data handling

#### [`ImagesDataset`](torchio/data/images.py)

`ImagesDataset` is a reader of 3D medical images that directly inherits from
[`torch.utils.Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset).
It can be used with a
[`torch.utils.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
for efficient loading and data augmentation.

It receives a list of subjects, where each subject is an instance of
[`torchio.Subject`](torchio/data/images.py) containing instances of
[`torchio.Image`](torchio/data/images.py).
The file format must be compatible with [NiBabel](https://nipy.org/nibabel/) or
[SimpleITK](http://www.simpleitk.org/) readers.
It can also be a directory containing
[DICOM](https://www.dicomstandard.org/) files.

```python
import torchio
from torchio import ImagesDataset, Image, Subject

subject_a = Subject([
    Image('t1', '~/Dropbox/MRI/t1.nrrd', torchio.INTENSITY),
    Image('label', '~/Dropbox/MRI/t1_seg.nii.gz', torchio.LABEL),
])
subject_b = Subject(
    Image('t1', '/tmp/colin27_t1_tal_lin.nii.gz', torchio.INTENSITY),
    Image('t2', '/tmp/colin27_t2_tal_lin.nii', torchio.INTENSITY),
    Image('label', '/tmp/colin27_seg1.nii.gz', torchio.LABEL),
)
subjects_list = [subject_a, subject_b]
subjects_dataset = ImagesDataset(subjects_list)
subject_sample = subjects_dataset[0]
```


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


### Transforms

The transforms package should remind users of
[`torchvision.transforms`](https://pytorch.org/docs/stable/torchvision/transforms.html).
They take as input the samples generated by an [`ImagesDataset`](#dataset).

A transform can be quickly applied to an image file using the command-line
tool `torchio-transform`:

```shell
$ torchio-transform input.nii.gz RandomMotion output.nii.gz --kwargs "proportion_to_augment=1 num_transforms=4"
```

#### Augmentation

##### Intensity

###### [MRI k-space motion artifacts](torchio/transforms/augmentation/intensity/random_motion.py)

Magnetic resonance images suffer from motion artifacts when the subject moves
during image acquisition. This transform follows
[Shaw et al., 2019](http://proceedings.mlr.press/v102/shaw19a.html) to
simulate motion artifacts for data augmentation.

![MRI k-space motion artifacts](https://raw.githubusercontent.com/fepegar/torchio/master/images/random_motion.gif)


###### [MRI k-space ghosting artifacts](torchio/transforms/augmentation/intensity/random_ghosting.py)

Discrete "ghost" artifacts may occur along the phase-encode direction whenever the position or signal intensity of imaged structures within the field-of-view vary or move in a regular (periodic) fashion.
Pulsatile flow of blood or CSF, cardiac motion, and respiratory motion are the most important patient-related causes of ghost artifacts in clinical MR imaging (From [mriquestions.com](http://mriquestions.com/why-discrete-ghosts.html)).

![MRI k-space ghosting artifacts](https://raw.githubusercontent.com/fepegar/torchio/master/images/random_ghosting.gif)


###### [MRI k-space spike artifacts](torchio/transforms/augmentation/intensity/random_spike.py)

Also known as [Herringbone artifact](https://radiopaedia.org/articles/herringbone-artifact?lang=gb), crisscross artifact or corduroy artifact,
it creates stripes in different directions in image space due to spikes in k-space.

![MRI k-space spike artifacts](https://raw.githubusercontent.com/fepegar/torchio/master/images/random_spike.gif)


###### [MRI magnetic field inhomogeneity](torchio/transforms/augmentation/intensity/random_bias_field.py)

MRI magnetic field inhomogeneity creates slow frequency intensity variations.
This transform is very similar to the one in
[NiftyNet](https://niftynet.readthedocs.io/en/dev/niftynet.layer.rand_bias_field.html).

![MRI bias field artifacts](https://raw.githubusercontent.com/fepegar/torchio/master/images/random_bias_field.gif)


##### [Patch swap](torchio/transforms/augmentation/intensity/random_swap.py)

Randomly swaps patches in the image.
This is typically used in
[context restoration for self-supervised learning](https://www.sciencedirect.com/science/article/pii/S1361841518304699).

![Random patches swapping](https://raw.githubusercontent.com/fepegar/torchio/master/images/random_swap.jpg)


###### [Gaussian noise](torchio/transforms/augmentation/intensity/random_noise.py)

Adds noise sampled from a normal distribution with mean 0 and standard
deviation sampled from a uniform distribution in the range `std_range`.
It is often used after [`ZNormalization`](#z-normalization), as the output of
this transform has zero-mean.

![Random Gaussian noise](https://raw.githubusercontent.com/fepegar/torchio/master/images/random_noise.gif)


###### [Gaussian blurring](torchio/transforms/augmentation/intensity/random_blur.py)

Blurs the image using a
[discrete Gaussian image filter](https://itk.org/Doxygen/html/classitk_1_1DiscreteGaussianImageFilter.html).


##### Spatial

###### [B-spline dense elastic deformation](torchio/transforms/augmentation/spatial/random_elastic_deformation.py)
<p align="center">
  <img src="https://raw.githubusercontent.com/fepegar/torchio/master/images/random_elastic_deformation.gif" alt="Random elastic deformation"/>
</p>


###### [Flip](torchio/transforms/augmentation/spatial/random_flip.py)

Reverse the order of elements in an image along the given axes.


###### [Affine transform](torchio/transforms/augmentation/spatial/random_affine.py)

Random affine transformation of the image keeping center invariant.


#### Preprocessing

##### [Histogram standardization](torchio/transforms/preprocessing/intensity/histogram_standardization.py)

Implementation of
[*New variants of a method of MRI scale standardization*](https://ieeexplore.ieee.org/document/836373)
adapted from NiftyNet.

![Histogram standardization](https://raw.githubusercontent.com/fepegar/torchio/master/images/histogram_standardization.png)


##### [Rescale](torchio/transforms/preprocessing/intensity/rescale.py)

Rescale intensity values in an image to a certain range.


##### [Z-normalization](torchio/transforms/preprocessing/intensity/z_normalization.py)

This transform first extracts the values with intensity greater than the mean,
which is an approximation of the foreground voxels.
Then the foreground mean is subtracted from the image and it is divided by the
foreground standard deviation.


##### [Resample](torchio/transforms/preprocessing/spatial/resample.py)

Resample images to a new voxel spacing using `nibabel`.


##### [Pad](torchio/transforms/preprocessing/spatial/pad.py)

Pad images, like in [`torchvision.transforms.Pad`](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Pad).


##### [Crop](torchio/transforms/preprocessing/spatial/crop.py)

Crop images passing 1, 3, or 6 integers, as in [Pad](#pad).


##### [ToCanonical](torchio/transforms/preprocessing/spatial/to_canonical.py)

Reorder the data so that it is closest to canonical NIfTI (RAS+) orientation.


##### [CenterCropOrPad](torchio/transforms/preprocessing/spatial/center_crop_pad.py)

Crops or pads image center to a target size, modifying the affine accordingly.


#### Others

##### [Lambda](torchio/transforms/lambda_transform.py)

Applies a user-defined function as transform.
For example, image intensity can be inverted with
`Lambda(lambda x: -x, types_to_apply=[torchio.INTENSITY])`
and a mask can be negated with
`Lambda(lambda x: 1 - x, types_to_apply=[torchio.LABEL])`.



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
