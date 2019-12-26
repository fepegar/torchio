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
    subject_dict = {
        'T1': dict(path='/tmp/mni/t1_on_mni.nii.gz', type=torchio.INTENSITY),
        'label': dict(path='/tmp/mni/t1_259_resection_seg.nii.gz', type=torchio.LABEL),
    }
    subjects_paths = [subject_dict]  # just one

    verbose = True

    scales = (0.75, 0.75)
    degrees = (-5, -5)
    axes = (0,)

    transforms = (
        RandomAffine(scales=scales, degrees=degrees,
                     isotropic=False, verbose=verbose),
        RandomFlip(axes, verbose=verbose),
    )
    transform = Compose(transforms)
    subjects_dataset = ImagesDataset(
        subjects_paths, transform=transform, verbose=verbose)
    sample = subjects_dataset[0]
    save_sample(sample, '/tmp/sample')
