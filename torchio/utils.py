import ast
import shutil
import tempfile
from pathlib import Path
from typing import Union, Iterable, Tuple, Any, Optional, List

import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import trange
from .torchio import (
    INTENSITY,
    LABEL,
    TypeData,
    TypeNumber,
    TypePath,
)


FLIP_XY = np.diag((-1, -1, 1))  # used to switch between LPS and RAS


def to_tuple(
        value: Union[TypeNumber, Iterable[TypeNumber]],
        length: int = 1,
        ) -> Tuple[TypeNumber, ...]:
    """
    to_tuple(1, length=1) -> (1,)
    to_tuple(1, length=3) -> (1, 1, 1)

    If value is an iterable, n is ignored and tuple(value) is returned
    to_tuple((1,), length=1) -> (1,)
    to_tuple((1, 2), length=1) -> (1, 2)
    to_tuple([1, 2], length=3) -> (1, 2)
    """
    try:
        iter(value)
        value = tuple(value)
    except TypeError:
        value = length * (value,)
    return value


def get_stem(path: TypePath) -> str:
    """
    '/home/user/image.nii.gz' -> 'image'
    """
    path = Path(path)
    return path.name.split('.')[0]


def create_dummy_dataset(
        num_images: int,
        size_range: Tuple[int, int],
        directory: Optional[TypePath] = None,
        suffix: str = '.nii.gz',
        force: bool = False,
        verbose: bool = False,
        ):
    from .data import Image, Subject
    output_dir = tempfile.gettempdir() if directory is None else directory
    output_dir = Path(output_dir)
    images_dir = output_dir / 'dummy_images'
    labels_dir = output_dir / 'dummy_labels'

    if force:
        shutil.rmtree(images_dir)
        shutil.rmtree(labels_dir)

    subjects: List[Subject] = []
    if images_dir.is_dir():
        for i in trange(num_images):
            image_path = images_dir / f'image_{i}{suffix}'
            label_path = labels_dir / f'label_{i}{suffix}'
            subject = Subject(
                one_modality=Image(image_path, INTENSITY),
                segmentation=Image(label_path, LABEL),
            )
            subjects.append(subject)
    else:
        images_dir.mkdir(exist_ok=True, parents=True)
        labels_dir.mkdir(exist_ok=True, parents=True)
        if verbose:
            print('Creating dummy dataset...')
            iterable = trange(num_images)
        else:
            iterable = range(num_images)
        for i in iterable:
            shape = np.random.randint(*size_range, size=3)
            affine = np.eye(4)
            image = np.random.rand(*shape)
            label = np.ones_like(image)
            label[image < 0.33] = 0
            label[image > 0.66] = 2
            image *= 255

            image_path = images_dir / f'image_{i}{suffix}'
            nii = nib.Nifti1Image(image.astype(np.uint8), affine)
            nii.to_filename(str(image_path))

            label_path = labels_dir / f'label_{i}{suffix}'
            nii = nib.Nifti1Image(label.astype(np.uint8), affine)
            nii.to_filename(str(label_path))

            subject = Subject(
                one_modality=Image(image_path, INTENSITY),
                segmentation=Image(label_path, LABEL),
            )
            subjects.append(subject)
    return subjects


def apply_transform_to_file(
        input_path: TypePath,
        transform,  # : Transform seems to create a circular import (TODO)
        output_path: TypePath,
        type: str = INTENSITY,
        ):
    from . import Image, ImagesDataset, Subject
    subject = Subject(image=Image(input_path, type))
    dataset = ImagesDataset([subject], transform=transform)
    transformed = dataset[0]
    dataset.save_sample(transformed, dict(image=output_path))


def guess_type(string: str) -> Any:
    # Adapted from
    # https://www.reddit.com/r/learnpython/comments/4599hl/module_to_guess_type_from_a_string/czw3f5s
    string = string.replace(' ', '')
    try:
        value = ast.literal_eval(string)
    except ValueError:
        result_type = str
    else:
        result_type = type(value)
    if result_type in (list, tuple):
        string = string[1:-1]  # remove brackets
        split = string.split(',')
        list_result = [guess_type(n) for n in split]
        value = tuple(list_result) if result_type is tuple else list_result
        return value
    try:
        value = result_type(string)
    except TypeError:
        value = None
    return value


def get_rotation_and_spacing_from_affine(
        affine: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
    # From https://github.com/nipy/nibabel/blob/master/nibabel/orientations.py
    rotation_zoom = affine[:3, :3]
    spacing = np.sqrt(np.sum(rotation_zoom * rotation_zoom, axis=0))
    rotation = rotation_zoom / spacing
    return rotation, spacing


def nib_to_sitk(data: TypeData, affine: TypeData) -> sitk.Image:
    array = data.numpy() if isinstance(data, torch.Tensor) else data
    affine = affine.numpy() if isinstance(affine, torch.Tensor) else affine
    origin = np.dot(FLIP_XY, affine[:3, 3]).astype(np.float64)
    rotation, spacing = get_rotation_and_spacing_from_affine(affine)
    direction = np.dot(FLIP_XY, rotation)
    image = sitk.GetImageFromArray(array.transpose())
    if array.ndim == 2:  # ignore first dimension if 2D (1, 1, H, W)
        direction = direction[1:3, 1:3]
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction.flatten())
    return image


def sitk_to_nib(image: sitk.Image) -> Tuple[np.ndarray, np.ndarray]:
    data = sitk.GetArrayFromImage(image).transpose()
    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection())
    origin = image.GetOrigin()
    if len(direction) == 9:
        rotation = direction.reshape(3, 3)
    elif len(direction) == 4:  # ignore first dimension if 2D (1, 1, H, W)
        rotation_2d = direction.reshape(2, 2)
        rotation = np.eye(3)
        rotation[1:3, 1:3] = rotation_2d
        spacing = 1, *spacing
        origin = 0, *origin
    rotation = np.dot(FLIP_XY, rotation)
    rotation_zoom = rotation * spacing
    translation = np.dot(FLIP_XY, origin)
    affine = np.eye(4)
    affine[:3, :3] = rotation_zoom
    affine[:3, 3] = translation
    return data, affine


def get_torchio_cache_dir():
    return Path('~/.cache/torchio').expanduser()


def round_up(value: float) -> float:
    """Round half towards infinity.

    Args:
        value: The value to round.

    Example:

        >>> round(2.5)
        2
        >>> round(3.5)
        4
        >>> round_up(2.5)
        3
        >>> round_up(3.5)
        4

    """
    return np.floor(value + 0.5)
