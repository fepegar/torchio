import ast
import gzip
import shutil
import tempfile
from pathlib import Path
from typing import Union, Iterable, Tuple, Any, Optional, List, Sequence

from torch.utils.data._utils.collate import default_collate
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import trange

from .typing import TypeNumber, TypePath


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
            print('Creating dummy dataset...')  # noqa: T001
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
        transform,  # : Transform seems to create a circular import
        output_path: TypePath,
        class_: str = 'ScalarImage',
        verbose: bool = False,
        ):
    from . import data
    image = getattr(data, class_)(input_path)
    subject = data.Subject(image=image)
    transformed = transform(subject)
    transformed.image.save(output_path)
    if verbose and transformed.history:
        print('Applied transform:', transformed.history[0])  # noqa: T001


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


def get_torchio_cache_dir():
    return Path('~/.cache/torchio').expanduser()


def round_up(value: float) -> int:
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
    return int(np.floor(value + 0.5))


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


def get_major_sitk_version() -> int:
    # This attribute was added in version 2
    # https://github.com/SimpleITK/SimpleITK/pull/1171
    version = getattr(sitk, '__version__', None)
    major_version = 1 if version is None else 2
    return major_version


def history_collate(batch: Sequence, collate_transforms=True):
    attr = 'history' if collate_transforms else 'applied_transforms'
    # Adapted from
    # https://github.com/romainVala/torchQC/blob/master/segmentation/collate_functions.py
    from .data import Subject
    first_element = batch[0]
    if isinstance(first_element, Subject):
        dictionary = {
            key: default_collate([d[key] for d in batch])
            for key in first_element
        }
        if hasattr(first_element, attr):
            dictionary.update({attr: [getattr(d, attr) for d in batch]})
        return dictionary


def get_subclasses(target_class: type) -> List[type]:
    subclasses = target_class.__subclasses__()
    subclasses += sum((get_subclasses(cls) for cls in subclasses), [])
    return subclasses
