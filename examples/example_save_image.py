from pathlib import Path
from itertools import islice

import numpy as np
import nibabel as nib

import torch
from torchvision.transforms import Compose

from torchio import ImagesDataset
from torchio.transforms import RandomFlip, RandomAffine, Interpolation


def save_sample(sample, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    for key in 'image', 'label':
        sample_array = sample[key]
        path = output_dir / f'{key}.nii.gz'
        sample_array = sample_array.squeeze()
        nii = nib.Nifti1Image(sample_array, np.eye(4))
        nii.to_filename(str(path))


if __name__ == "__main__":
    torch.manual_seed(42)

    # Config
    force = False

    paths_dict = dict(
        image=['/tmp/mni/t1_on_mni.nii.gz'],
        label=['/tmp/mni/t1_259_resection_seg.nii.gz'],
    )

    verbose = True

    scales = (0.75, 0.75)
    angles = (-5, -5)
    axes = (0,)

    transforms = (
        RandomAffine(scales=scales, angles=angles,
                     isotropic=False, verbose=verbose),
        RandomFlip(axes, verbose=verbose),
    )
    transform = Compose(transforms)
    subjects_dataset = ImagesDataset(
        paths_dict, transform=transform, verbose=verbose)
    sample = subjects_dataset[0]
    save_sample(sample, '/tmp/sample')
