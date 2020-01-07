import shutil
import tempfile
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import trange
from .torchio import INTENSITY, LABEL


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
    """
    '/home/user/image.nii.gz' -> 'image'
    """
    path = Path(path)
    return path.name.split('.')[0]


def is_image_dict(variable):
    is_dict = isinstance(variable, dict)
    if not is_dict:
        return False
    has_right_keys = (
        'type' in variable
        and 'data' in variable
        and 'affine' in variable
    )
    return has_right_keys


def create_dummy_dataset(num_images, size_range, force=False):
    tempdir = Path(tempfile.gettempdir())
    images_dir = tempdir / 'dummy_images'
    labels_dir = tempdir / 'dummy_labels'

    if force:
        shutil.rmtree(images_dir)
        shutil.rmtree(labels_dir)

    subjects_paths = []
    if images_dir.is_dir():
        for i in trange(num_images):
            image_path = images_dir / f'image_{i}.nii.gz'
            label_path = labels_dir / f'label_{i}.nii.gz'
            paths_dict = dict(
                one_modality=dict(type=INTENSITY, path=image_path),
                segmentation=dict(type=LABEL, path=image_path),
            )
            subjects_paths.append(paths_dict)
    else:
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

            image_path = images_dir / f'image_{i}.nii.gz'
            nii = nib.Nifti1Image(image.astype(np.uint8), affine)
            nii.to_filename(str(image_path))

            label_path = labels_dir / f'label_{i}.nii.gz'
            nii = nib.Nifti1Image(label.astype(np.uint8), affine)
            nii.to_filename(str(label_path))

            paths_dict = dict(
                one_modality=dict(type=INTENSITY, path=image_path),
                segmentation=dict(type=LABEL, path=image_path),
            )
            subjects_paths.append(paths_dict)
    return subjects_paths
