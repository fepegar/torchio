import ast
import shutil
import pprint
import tempfile
from pathlib import Path
from typing import Union, Iterable, Tuple, Any, Optional
import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import trange
from .torchio import (
    INTENSITY, LABEL, DATA, AFFINE, TYPE, TypeData, TypeNumber, TypePath)


FLIP_XY = np.diag((-1, -1, 1))


def to_tuple(
        value: Union[TypeNumber, Iterable[TypeNumber]],
        n: int = 1,
        ) -> Tuple[TypeNumber, ...]:
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


def get_stem(path: TypePath) -> str:
    """
    '/home/user/image.nii.gz' -> 'image'
    """
    path = Path(path)
    return path.name.split('.')[0]


def is_image_dict(variable: Any) -> bool:
    is_dict = isinstance(variable, dict)
    if not is_dict:
        return False
    has_right_keys = (
        TYPE in variable
        and DATA in variable
        and AFFINE in variable
    )
    return has_right_keys


def create_dummy_dataset(
        num_images: int,
        size_range: Tuple[int, int],
        directory: Optional[TypePath] = None,
        suffix: str = '.nii.gz',
        force: bool = False,
        verbose: bool = False,
        ):
    from .data import Image
    output_dir = tempfile.gettempdir() if directory is None else directory
    output_dir = Path(output_dir)
    images_dir = output_dir / 'dummy_images'
    labels_dir = output_dir / 'dummy_labels'

    if force:
        shutil.rmtree(images_dir)
        shutil.rmtree(labels_dir)

    subjects = []
    if images_dir.is_dir():
        for i in trange(num_images):
            image_path = images_dir / f'image_{i}{suffix}'
            label_path = labels_dir / f'label_{i}{suffix}'
            subject_images = [
                Image('one_modality', image_path, INTENSITY),
                Image('segmentation', label_path, LABEL),
            ]
            subjects.append(subject_images)
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

            subject_images = [
                Image('one_modality', image_path, INTENSITY),
                Image('segmentation', label_path, LABEL),
            ]
            subjects.append(subject_images)
    return subjects


def apply_transform_to_file(
        input_path: TypePath,
        transform,  # : Transform seems to create a circular import (TODO)
        output_path: TypePath,
        type_: str = INTENSITY,
        ):
    from . import Image, ImagesDataset
    subject = [
        Image('image', input_path, type_),
    ]
    dataset = ImagesDataset([subject], transform=transform)
    transformed = dataset[0]
    dataset.save_sample(transformed, dict(image=output_path))


def guess_type(string: str) -> Any:
    """
    Adapted from
    https://www.reddit.com/r/learnpython/comments/4599hl/module_to_guess_type_from_a_string/czw3f5s
    """
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


def check_consistent_shape(sample: dict) -> None:
    shapes_dict = {}
    for image_name, image_dict in sample.items():
        if not is_image_dict(image_dict):
            continue
        shapes_dict[image_name] = image_dict[DATA].shape
    num_unique_shapes = len(set(shapes_dict.values()))
    if num_unique_shapes > 1:
        message = (
            'Images in sample have inconsistent shapes:'
            f'\n{pprint.pformat(shapes_dict)}'
        )
        raise ValueError(message)


def get_rotation_and_spacing_from_affine(
        affine: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
    RZS = affine[:3, :3]
    spacing = np.sqrt(np.sum(RZS * RZS, axis=0))
    R = RZS / spacing
    return R, spacing


def nib_to_sitk(data: TypeData, affine: TypeData) -> sitk.Image:
    array = data.numpy() if isinstance(data, torch.Tensor) else data
    affine = affine.numpy() if isinstance(affine, torch.Tensor) else affine
    origin = np.dot(FLIP_XY, affine[:3, 3]).astype(np.float64)
    R, spacing = get_rotation_and_spacing_from_affine(affine)
    direction = np.dot(FLIP_XY, R).flatten()
    image = sitk.GetImageFromArray(array.transpose())
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    return image


def sitk_to_nib(image: sitk.Image) -> Tuple[np.ndarray, np.ndarray]:
    data = sitk.GetArrayFromImage(image).transpose()
    spacing = np.array(image.GetSpacing())
    R = np.array(image.GetDirection()).reshape(3, 3)
    R = np.dot(FLIP_XY, R)
    RZS = R * spacing
    translation = np.dot(FLIP_XY, image.GetOrigin())
    affine = np.eye(4)
    affine[:3, :3] = RZS
    affine[:3, 3] = translation
    return data, affine
