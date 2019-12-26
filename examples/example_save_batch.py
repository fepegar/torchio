from pathlib import Path
from itertools import islice

import numpy as np
import nibabel as nib

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

import torchio
from torchio import ImagesDataset
from torchio.sampler import LabelSampler
from torchio.transforms import RandomFlip, RandomAffine, Interpolation


def save_batch(batch, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    for key in 'image', 'label':
        for i, sample_array in enumerate(batch[key]):
            path = output_dir / f'{key}_{i}.nii.gz'
            sample_array = sample_array.numpy().squeeze()
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
    patch_size = 128

    batch_size = 4

    scales = (0.75, 0.75)
    degrees = (-5, -5)
    axes = (0,)

    transforms = (
        RandomAffine(scales=scales, degrees=degrees, image_interpolation=Interpolation.BSPLINE,
                     isotropic=False, verbose=verbose),
        RandomFlip(axes, verbose=verbose),
    )
    transform = Compose(transforms)
    subjects_dataset = ImagesDataset(
        subjects_paths, transform=transform, verbose=verbose)
    sample = subjects_dataset[0]

    sampler = LabelSampler(sample, patch_size)
    loader = DataLoader(sampler, batch_size=batch_size)

    # TODO: check that this works as expected, use dummy data
    for batch in islice(loader, 1):
        save_batch(batch, '/tmp/batch')
