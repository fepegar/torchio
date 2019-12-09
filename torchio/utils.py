import shutil
import tempfile
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import trange


def to_tuple(value, n=1):
    """
    to_tuple(1, n=1) -> (1,)
    to_tuple(1, n=3) -> (1, 1, 1)

    If value is an iterable, n is ignored and tuple(value) is returned
    to_tuple((1,), n=1) -> (1,)
    to_tuple((1, 2), n=1) -> (1, 2)
    to_tuple([1, 2], n=3) -> (1, 2)
    """
    try:
        iter(value)
        value = tuple(value)
    except TypeError:
        value = n * (value,)
    return value


def get_stem(path):
    path = Path(path)
    return path.name.split('.')[0]


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
        print('Creating dummy dataset...')
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
