import ast
import gzip
import shutil
import tempfile
from pathlib import Path
from typing import Union, Iterable, Tuple, Any, Optional, List, Sequence

import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import trange
from .torchio import (
    INTENSITY,
    TypeData,
    TypeNumber,
    TypePath,
    REPO_URL,
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


def get_stem(
        path: Union[TypePath, List[TypePath]]
        ) -> Union[str, List[str]]:
    """
    '/home/user/image.nii.gz' -> 'image'
    """
    def _get_stem(path_string):
        return Path(path_string).name.split('.')[0]
    if isinstance(path, (str, Path)):
        return _get_stem(path)
    return [_get_stem(p) for p in path]


def create_dummy_dataset(
        num_images: int,
        size_range: Tuple[int, int],
        directory: Optional[TypePath] = None,
        suffix: str = '.nii.gz',
        force: bool = False,
        verbose: bool = False,
        ):
    from .data import ScalarImage, LabelMap, Subject
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
                one_modality=ScalarImage(image_path),
                segmentation=LabelMap(label_path),
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
                one_modality=ScalarImage(image_path),
                segmentation=LabelMap(label_path),
            )
            subjects.append(subject)
    return subjects


def apply_transform_to_file(
        input_path: TypePath,
        transform,  # : Transform seems to create a circular import (TODO)
        output_path: TypePath,
        type: str = INTENSITY,
        verbose: bool = False,
        ):
    from . import Image, SubjectsDataset, Subject
    subject = Subject(image=Image(input_path, type=type))
    transformed = transform(subject)
    transformed.image.save(output_path)
    if verbose and transformed.history:
        print(transformed.history[0])


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


def nib_to_sitk(
        data: TypeData,
        affine: TypeData,
        squeeze: bool = False,
        force_3d: bool = False,
        force_4d: bool = False,
        ) -> sitk.Image:
    """Create a SimpleITK image from a tensor and a 4x4 affine matrix.

    Args:
        data: PyTorch tensor or NumPy array
        affine: # TODO
    """
    if data.ndim != 4:
        raise ValueError(f'Input must be 4D, but has shape {tuple(data.shape)}')
    # Possibilities
    # (1, w, h, 1)
    # (c, w, h, 1)
    # (1, w, h, 1)
    # (c, w, h, d)
    array = np.asarray(data)
    affine = np.asarray(affine).astype(np.float64)

    is_multichannel = array.shape[0] > 1 and not force_4d
    is_2d = array.shape[3] == 1 and not force_3d
    if is_2d:
        array = array[..., 0]
    if not is_multichannel and not force_4d:
        array = array[0]
    array = array.transpose()  # (W, H, D, C) or (W, H, D)
    image = sitk.GetImageFromArray(array, isVector=is_multichannel)

    rotation, spacing = get_rotation_and_spacing_from_affine(affine)
    origin = np.dot(FLIP_XY, affine[:3, 3])
    direction = np.dot(FLIP_XY, rotation)
    if is_2d:  # ignore first dimension if 2D (1, W, H, 1)
        direction = direction[:2, :2]
    image.SetOrigin(origin)  # should I add a 4th value if force_4d?
    image.SetSpacing(spacing)
    image.SetDirection(direction.flatten())
    if data.ndim == 4:
        assert image.GetNumberOfComponentsPerPixel() == data.shape[0]
    num_spatial_dims = 2 if is_2d else 3
    assert image.GetSize() == data.shape[1: 1 + num_spatial_dims]
    return image


def sitk_to_nib(
        image: sitk.Image,
        keepdim: bool = False,
        ) -> Tuple[np.ndarray, np.ndarray]:
    data = sitk.GetArrayFromImage(image).transpose()
    num_components = image.GetNumberOfComponentsPerPixel()
    if num_components == 1:
        data = data[np.newaxis]  # add channels dimension
    input_spatial_dims = image.GetDimension()
    if input_spatial_dims == 2:
        data = data[..., np.newaxis]
    if not keepdim:
        data = ensure_4d(data, num_spatial_dims=input_spatial_dims)
    assert data.shape[0] == num_components
    assert data.shape[1: 1 + input_spatial_dims] == image.GetSize()
    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection())
    origin = image.GetOrigin()
    if len(direction) == 9:
        rotation = direction.reshape(3, 3)
    elif len(direction) == 4:  # ignore first dimension if 2D (1, W, H, 1)
        rotation_2d = direction.reshape(2, 2)
        rotation = np.eye(3)
        rotation[:2, :2] = rotation_2d
        spacing = *spacing, 1
        origin = *origin, 0
    else:
        raise RuntimeError(f'Direction not understood: {direction}')
    rotation = np.dot(FLIP_XY, rotation)
    rotation_zoom = rotation * spacing
    translation = np.dot(FLIP_XY, origin)
    affine = np.eye(4)
    affine[:3, :3] = rotation_zoom
    affine[:3, 3] = translation
    return data, affine


def ensure_4d(tensor: TypeData, num_spatial_dims=None) -> TypeData:
    # I wish named tensors were properly supported in PyTorch
    num_dimensions = tensor.ndim
    if num_dimensions == 4:
        pass
    elif num_dimensions == 5:  # hope (X, X, X, 1, X)
        if tensor.shape[-1] == 1:
            tensor = tensor[..., 0, :]
    elif num_dimensions == 2:  # assume 2D monochannel (W, H)
        tensor = tensor[np.newaxis, ..., np.newaxis]  # (1, W, H, 1)
    elif num_dimensions == 3:  # 2D multichannel or 3D monochannel?
        if num_spatial_dims == 2:
            tensor = tensor[..., np.newaxis]  # (C, W, H, 1)
        elif num_spatial_dims == 3:  # (W, H, D)
            tensor = tensor[np.newaxis]  # (1, W, H, D)
        else:  # try to guess
            shape = tensor.shape
            maybe_rgb = 3 in (shape[0], shape[-1])
            if maybe_rgb:
                if shape[-1] == 3:  # (W, H, 3)
                    tensor = tensor.permute(2, 0, 1)  # (3, W, H)
                tensor = tensor[..., np.newaxis]  # (3, W, H, 1)
            else:  # (W, H, D)
                tensor = tensor[np.newaxis]  # (1, W, H, D)
    else:
        message = (
            f'{num_dimensions}D images not supported yet. Please create an'
            f' issue in {REPO_URL} if you would like support for them'
        )
        raise ValueError(message)
    assert tensor.ndim == 4
    return tensor


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


def compress(input_path, output_path):
    with open(input_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def check_sequence(sequence: Sequence, name: str):
    try:
        iter(sequence)
    except TypeError:
        message = f'"{name}" must be a sequence, not {type(name)}'
        raise TypeError(message)
