import warnings
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Union, Sequence, List, Callable

import torch
import humanize
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from deprecated import deprecated

from ..utils import get_stem
from ..typing import TypeData, TypePath, TypeTripletInt, TypeTripletFloat
from ..constants import DATA, TYPE, AFFINE, PATH, STEM, INTENSITY, LABEL
from .io import (
    ensure_4d,
    read_image,
    write_image,
    nib_to_sitk,
    check_uint_to_int,
    get_rotation_and_spacing_from_affine,
)


PROTECTED_KEYS = DATA, AFFINE, TYPE, PATH, STEM
TypeBound = Tuple[float, float]
TypeBounds = Tuple[TypeBound, TypeBound, TypeBound]

deprecation_message = (
    'Setting the image data with the property setter is deprecated. Use the'
    ' set_data() method instead'
)


class Image(dict):
    r"""TorchIO image.

    For information about medical image orientation, check out `NiBabel docs`_,
    the `3D Slicer wiki`_, `Graham Wideman's website`_, `FSL docs`_ or
    `SimpleITK docs`_.

    Args:
        path: Path to a file or sequence of paths to files that can be read by
            :mod:`SimpleITK` or :mod:`nibabel`, or to a directory containing
            DICOM files. If :attr:`tensor` is given, the data in
            :attr:`path` will not be read.
            If a sequence of paths is given, data
            will be concatenated on the channel dimension so spatial
            dimensions must match.
        type: Type of image, such as :attr:`torchio.INTENSITY` or
            :attr:`torchio.LABEL`. This will be used by the transforms to
            decide whether to apply an operation, or which interpolation to use
            when resampling. For example, `preprocessing`_ and `augmentation`_
            intensity transforms will only be applied to images with type
            :attr:`torchio.INTENSITY`. Spatial transforms will be applied to
            all types, and nearest neighbor interpolation is always used to
            resample images with type :attr:`torchio.LABEL`.
            The type :attr:`torchio.SAMPLING_MAP` may be used with instances of
            :class:`~torchio.data.sampler.weighted.WeightedSampler`.
        tensor: If :attr:`path` is not given, :attr:`tensor` must be a 4D
            :class:`torch.Tensor` or NumPy array with dimensions
            :math:`(C, W, H, D)`.
        affine: If :attr:`path` is not given, :attr:`affine` must be a
            :math:`4 \times 4` NumPy array. If ``None``, :attr:`affine` is an
            identity matrix.
        check_nans: If ``True``, issues a warning if NaNs are found
            in the image. If ``False``, images will not be checked for the
            presence of NaNs.
        reader: Callable object that takes a path and returns a 4D tensor and a
            2D, :math:`4 \times 4` affine matrix. This can be used if your data
            is saved in a custom format, such as ``.npy`` (see example below).
            If the affine matrix is ``None``, an identity matrix will be used.
        **kwargs: Items that will be added to the image dictionary, e.g.
            acquisition parameters.

    TorchIO images are `lazy loaders`_, i.e. the data is only loaded from disk
    when needed.

    Example:
        >>> import torchio as tio
        >>> import numpy as np
        >>> image = tio.ScalarImage('t1.nii.gz')  # subclass of Image
        >>> image  # not loaded yet
        ScalarImage(path: t1.nii.gz; type: intensity)
        >>> times_two = 2 * image.data  # data is loaded and cached here
        >>> image
        ScalarImage(shape: (1, 256, 256, 176); spacing: (1.00, 1.00, 1.00); orientation: PIR+; memory: 44.0 MiB; type: intensity)
        >>> image.save('doubled_image.nii.gz')
        >>> numpy_reader = lambda path: np.load(path), np.eye(4)
        >>> image = tio.ScalarImage('t1.npy', reader=numpy_reader)

    .. _lazy loaders: https://en.wikipedia.org/wiki/Lazy_loading
    .. _preprocessing: https://torchio.readthedocs.io/transforms/preprocessing.html#intensity
    .. _augmentation: https://torchio.readthedocs.io/transforms/augmentation.html#intensity
    .. _NiBabel docs: https://nipy.org/nibabel/image_orientation.html
    .. _3D Slicer wiki: https://www.slicer.org/wiki/Coordinate_systems
    .. _FSL docs: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Orientation%20Explained
    .. _SimpleITK docs: https://simpleitk.readthedocs.io/en/master/fundamentalConcepts.html
    .. _Graham Wideman's website: http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm
    """
    def __init__(
            self,
            path: Union[TypePath, Sequence[TypePath], None] = None,
            type: str = None,
            tensor: Optional[TypeData] = None,
            affine: Optional[TypeData] = None,
            check_nans: bool = False,  # removed by ITK by default
            channels_last: bool = False,
            reader: Callable = read_image,
            **kwargs: Dict[str, Any],
            ):
        self.check_nans = check_nans
        self.channels_last = channels_last
        self.reader = reader

        if type is None:
            warnings.warn(
                'Not specifying the image type is deprecated and will be'
                ' mandatory in the future. You can probably use tio.ScalarImage'
                ' or tio.LabelMap instead', DeprecationWarning,
            )
            type = INTENSITY

        if path is None and tensor is None:
            raise ValueError('A value for path or tensor must be given')
        self._loaded = False

        tensor = self._parse_tensor(tensor)
        affine = self._parse_affine(affine)
        if tensor is not None:
            self.set_data(tensor)
            self.affine = affine
            self._loaded = True
        for key in PROTECTED_KEYS:
            if key in kwargs:
                message = f'Key "{key}" is reserved. Use a different one'
                raise ValueError(message)

        super().__init__(**kwargs)
        self.path = self._parse_path(path)

        self[PATH] = '' if self.path is None else str(self.path)
        self[STEM] = '' if self.path is None else get_stem(self.path)
        self[TYPE] = type

    def __repr__(self):
        properties = []
        if self._loaded:
            properties.extend([
                f'shape: {self.shape}',
                f'spacing: {self.get_spacing_string()}',
                f'orientation: {"".join(self.orientation)}+',
                f'memory: {humanize.naturalsize(self.memory, binary=True)}',
            ])
        else:
            properties.append(f'path: "{self.path}"')
        if self._loaded:
            properties.append(f'dtype: {self.data.type()}')
        properties = '; '.join(properties)
        string = f'{self.__class__.__name__}({properties})'
        return string

    def __getitem__(self, item):
        if item in (DATA, AFFINE):
            if item not in self:
                self.load()
        return super().__getitem__(item)

    def __array__(self):
        return self.data.numpy()

    def __copy__(self):
        kwargs = dict(
            tensor=self.data,
            affine=self.affine,
            type=self.type,
            path=self.path,
        )
        for key, value in self.items():
            if key in PROTECTED_KEYS: continue
            kwargs[key] = value  # should I copy? deepcopy?
        return self.__class__(**kwargs)

    @property
    def data(self) -> torch.Tensor:
        """Tensor data. Same as :class:`Image.tensor`."""
        return self[DATA]

    @data.setter
    @deprecated(version='0.18.16', reason=deprecation_message)
    def data(self, tensor: TypeData):
        self.set_data(tensor)

    def set_data(self, tensor: TypeData):
        """Store a 4D tensor in the :attr:`data` key and attribute.

        Args:
            tensor: 4D tensor with dimensions :math:`(C, W, H, D)`.
        """
        self[DATA] = self._parse_tensor(tensor, none_ok=False)

    @property
    def tensor(self) -> torch.Tensor:
        """Tensor data. Same as :class:`Image.data`."""
        return self.data

    @property
    def affine(self) -> np.ndarray:
        """Affine matrix to transform voxel indices into world coordinates."""
        return self[AFFINE]

    @affine.setter
    def affine(self, matrix):
        self[AFFINE] = self._parse_affine(matrix)

    @property
    def type(self) -> str:
        return self[TYPE]

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Tensor shape as :math:`(C, W, H, D)`."""
        return tuple(self.data.shape)

    @property
    def spatial_shape(self) -> TypeTripletInt:
        """Tensor spatial shape as :math:`(W, H, D)`."""
        return self.shape[1:]

    def check_is_2d(self) -> None:
        if not self.is_2d():
            message = f'Image is not 2D. Spatial shape: {self.spatial_shape}'
            raise RuntimeError(message)

    @property
    def height(self) -> int:
        """Image height, if 2D."""
        self.check_is_2d()
        return self.spatial_shape[1]

    @property
    def width(self) -> int:
        """Image width, if 2D."""
        self.check_is_2d()
        return self.spatial_shape[0]

    @property
    def orientation(self) -> Tuple[str, str, str]:
        """Orientation codes."""
        return nib.aff2axcodes(self.affine)

    @property
    def spacing(self) -> Tuple[float, float, float]:
        """Voxel spacing in mm."""
        _, spacing = get_rotation_and_spacing_from_affine(self.affine)
        return tuple(spacing)

    @property
    def itemsize(self):
        """Element size of the data type."""
        return self.data.element_size()

    @property
    def memory(self) -> float:
        """Number of Bytes that the tensor takes in the RAM."""
        return np.prod(self.shape) * self.itemsize

    @property
    def bounds(self) -> np.ndarray:
        """Position of centers of voxels in smallest and largest coordinates."""
        ini = 0, 0, 0
        fin = np.array(self.spatial_shape) - 1
        point_ini = nib.affines.apply_affine(self.affine, ini)
        point_fin = nib.affines.apply_affine(self.affine, fin)
        return np.array((point_ini, point_fin))

    def axis_name_to_index(self, axis: str) -> int:
        """Convert an axis name to an axis index.

        Args:
            axis: Possible inputs are ``'Left'``, ``'Right'``, ``'Anterior'``,
                ``'Posterior'``, ``'Inferior'``, ``'Superior'``. Lower-case
                versions and first letters are also valid, as only the first
                letter will be used.

        .. note:: If you are working with animals, you should probably use
            ``'Superior'``, ``'Inferior'``, ``'Anterior'`` and ``'Posterior'``
            for ``'Dorsal'``, ``'Ventral'``, ``'Rostral'`` and ``'Caudal'``,
            respectively.

        .. note:: If your images are 2D, you can use ``'Top'``, ``'Bottom'``,
            ``'Left'`` and ``'Right'``.
        """
        # Top and bottom are used for the vertical 2D axis as the use of
        # Height vs Horizontal might be ambiguous

        if not isinstance(axis, str):
            raise ValueError('Axis must be a string')
        axis = axis[0].upper()

        # Generally, TorchIO tensors are (C, W, H, D)
        if axis in 'TB':  # Top, Bottom
            return -2
        else:
            try:
                index = self.orientation.index(axis)
            except ValueError:
                index = self.orientation.index(self.flip_axis(axis))
            # Return negative indices so that it does not matter whether we
            # refer to spatial dimensions or not
            index = -3 + index
            return index

    # flake8: noqa: E701
    @staticmethod
    def flip_axis(axis: str) -> str:
        if axis == 'R': flipped_axis = 'L'
        elif axis == 'L': flipped_axis = 'R'
        elif axis == 'A': flipped_axis = 'P'
        elif axis == 'P': flipped_axis = 'A'
        elif axis == 'I': flipped_axis = 'S'
        elif axis == 'S': flipped_axis = 'I'
        elif axis == 'T': flipped_axis = 'B'
        elif axis == 'B': flipped_axis = 'T'
        else:
            values = ', '.join('LRPAISTB')
            message = f'Axis not understood. Please use one of: {values}'
            raise ValueError(message)
        return flipped_axis

    def get_spacing_string(self) -> str:
        strings = [f'{n:.2f}' for n in self.spacing]
        string = f'({", ".join(strings)})'
        return string

    def get_bounds(self) -> TypeBounds:
        """Get minimum and maximum world coordinates occupied by the image."""
        first_index = 3 * (-0.5,)
        last_index = np.array(self.spatial_shape) - 0.5
        first_point = nib.affines.apply_affine(self.affine, first_index)
        last_point = nib.affines.apply_affine(self.affine, last_index)
        array = np.array((first_point, last_point))
        bounds_x, bounds_y, bounds_z = array.T.tolist()
        return bounds_x, bounds_y, bounds_z

    @staticmethod
    def _parse_single_path(
            path: TypePath
            ) -> Path:
        try:
            path = Path(path).expanduser()
        except TypeError:
            message = (
                f'Expected type str or Path but found {path} with type'
                f' {type(path)} instead'
            )
            raise TypeError(message)
        except RuntimeError:
            message = (
                f'Conversion to path not possible for variable: {path}'
            )
            raise RuntimeError(message)

        if not (path.is_file() or path.is_dir()):   # might be a dir with DICOM
            raise FileNotFoundError(f'File not found: "{path}"')
        return path

    def _parse_path(
            self,
            path: Union[TypePath, Sequence[TypePath]]
            ) -> Optional[Union[Path, List[Path]]]:
        if path is None:
            return None
        if isinstance(path, (str, Path)):
            return self._parse_single_path(path)
        else:
            return [self._parse_single_path(p) for p in path]

    def _parse_tensor(
            self,
            tensor: TypeData,
            none_ok: bool = True,
            ) -> torch.Tensor:
        if tensor is None:
            if none_ok:
                return None
            else:
                raise RuntimeError('Input tensor cannot be None')
        if isinstance(tensor, np.ndarray):
            tensor = check_uint_to_int(tensor)
            tensor = torch.as_tensor(tensor)
        elif not isinstance(tensor, torch.Tensor):
            message = 'Input tensor must be a PyTorch tensor or NumPy array'
            raise TypeError(message)
        if tensor.ndim != 4:
            raise ValueError('Input tensor must be 4D')
        if tensor.dtype == torch.bool:
            tensor = tensor.to(torch.uint8)
        if self.check_nans and torch.isnan(tensor).any():
            warnings.warn(f'NaNs found in tensor', RuntimeWarning)
        return tensor

    def parse_tensor_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        return ensure_4d(tensor)

    @staticmethod
    def _parse_affine(affine: TypeData) -> np.ndarray:
        if affine is None:
            return np.eye(4)
        if isinstance(affine, torch.Tensor):
            affine = affine.numpy()
        if not isinstance(affine, np.ndarray):
            raise TypeError(f'Affine must be a NumPy array, not {type(affine)}')
        if affine.shape != (4, 4):
            raise ValueError(f'Affine shape must be (4, 4), not {affine.shape}')
        return affine

    def load(self) -> None:
        r"""Load the image from disk.

        Returns:
            Tuple containing a 4D tensor of size :math:`(C, W, H, D)` and a 2D
            :math:`4 \times 4` affine matrix to convert voxel indices to world
            coordinates.
        """
        if self._loaded:
            return
        paths = self.path if isinstance(self.path, list) else [self.path]
        tensor, affine = self.read_and_check(paths[0])
        tensors = [tensor]
        for path in paths[1:]:
            new_tensor, new_affine = self.read_and_check(path)
            if not np.array_equal(affine, new_affine):
                message = (
                    'Files have different affine matrices.'
                    f'\nMatrix of {paths[0]}:'
                    f'\n{affine}'
                    f'\nMatrix of {path}:'
                    f'\n{new_affine}'
                )
                warnings.warn(message, RuntimeWarning)
            if not tensor.shape[1:] == new_tensor.shape[1:]:
                message = (
                    f'Files shape do not match, found {tensor.shape}'
                    f'and {new_tensor.shape}'
                )
                RuntimeError(message)
            tensors.append(new_tensor)
        tensor = torch.cat(tensors)
        self.set_data(tensor)
        self.affine = affine
        self._loaded = True

    def read_and_check(self, path: TypePath) -> Tuple[torch.Tensor, np.ndarray]:
        tensor, affine = self.reader(path)
        tensor = self.parse_tensor_shape(tensor)
        tensor = self._parse_tensor(tensor)
        affine = self._parse_affine(affine)
        if self.channels_last:
            tensor = tensor.permute(3, 0, 1, 2)
        if self.check_nans and torch.isnan(tensor).any():
            warnings.warn(f'NaNs found in file "{path}"', RuntimeWarning)
        return tensor, affine

    def save(self, path: TypePath, squeeze: bool = True) -> None:
        """Save image to disk.

        Args:
            path: String or instance of :class:`pathlib.Path`.
            squeeze: If ``True``, singleton dimensions will be removed
                before saving.
        """
        write_image(
            self.data,
            self.affine,
            path,
            squeeze=squeeze,
        )

    def is_2d(self) -> bool:
        return self.shape[-1] == 1

    def numpy(self) -> np.ndarray:
        """Get a NumPy array containing the image data."""
        return np.asarray(self)

    def as_sitk(self, **kwargs) -> sitk.Image:
        """Get the image as an instance of :class:`sitk.Image`."""
        return nib_to_sitk(self.data, self.affine, **kwargs)

    def as_pil(self, transpose=True):
        """Get the image as an instance of :class:`PIL.Image`.

        .. note:: Values will be clamped to 0-255 and cast to uint8.
        .. note:: To use this method, `Pillow` needs to be installed:
            `pip install Pillow`.
        """
        try:
            from PIL import Image as ImagePIL
        except ModuleNotFoundError as e:
            message = (
                'Please install Pillow to use Image.as_pil():'
                ' pip install Pillow'
            )
            raise RuntimeError(message) from e

        self.check_is_2d()
        tensor = self.data
        if len(tensor) == 1:
            tensor = torch.cat(3 * [tensor])
        if len(tensor) != 3:
            raise RuntimeError('The image must have 1 or 3 channels')
        if transpose:
            tensor = tensor.permute(3, 2, 1, 0)
        else:
            tensor = tensor.permute(3, 1, 2, 0)
        array = tensor.clamp(0, 255).numpy()[0]
        return ImagePIL.fromarray(array.astype(np.uint8))

    def get_center(self, lps: bool = False) -> TypeTripletFloat:
        """Get image center in RAS+ or LPS+ coordinates.

        Args:
            lps: If ``True``, the coordinates will be in LPS+ orientation, i.e.
                the first dimension grows towards the left, etc. Otherwise, the
                coordinates will be in RAS+ orientation.
        """
        size = np.array(self.spatial_shape)
        center_index = (size - 1) / 2
        r, a, s = nib.affines.apply_affine(self.affine, center_index)
        if lps:
            return (-r, -a, s)
        else:
            return (r, a, s)

    def set_check_nans(self, check_nans: bool) -> None:
        self.check_nans = check_nans

    def plot(self, **kwargs) -> None:
        """Plot image."""
        if self.is_2d():
            self.as_pil().show()
        else:
            from ..visualization import plot_volume  # avoid circular import
            plot_volume(self, **kwargs)


class ScalarImage(Image):
    """Image whose pixel values represent scalars.

    Example:
        >>> import torch
        >>> import torchio as tio
        >>> # Loading from a file
        >>> t1_image = tio.ScalarImage('t1.nii.gz')
        >>> dmri = tio.ScalarImage(tensor=torch.rand(32, 128, 128, 88))
        >>> image = tio.ScalarImage('safe_image.nrrd', check_nans=False)
        >>> data, affine = image.data, image.affine
        >>> affine.shape
        (4, 4)
        >>> image.data is image[tio.DATA]
        True
        >>> image.data is image.tensor
        True
        >>> type(image.data)
        torch.Tensor

    See :class:`~torchio.Image` for more information.
    """
    def __init__(self, *args, **kwargs):
        if 'type' in kwargs and kwargs['type'] != INTENSITY:
            raise ValueError('Type of ScalarImage is always torchio.INTENSITY')
        kwargs.update({'type': INTENSITY})
        super().__init__(*args, **kwargs)


class LabelMap(Image):
    """Image whose pixel values represent categorical labels.

    Intensity transforms are not applied to these images.

    Example:
        >>> import torch
        >>> import torchio as tio
        >>> labels = tio.LabelMap(tensor=torch.rand(1, 128, 128, 68) > 0.5)
        >>> labels = tio.LabelMap('t1_seg.nii.gz')  # loading from a file
        >>> tpm = tio.LabelMap(                     # loading from files
        ...     'gray_matter.nii.gz',
        ...     'white_matter.nii.gz',
        ...     'csf.nii.gz',
        ... )

    See :class:`~torchio.Image` for more information.
    """
    def __init__(self, *args, **kwargs):
        if 'type' in kwargs and kwargs['type'] != LABEL:
            raise ValueError('Type of LabelMap is always torchio.LABEL')
        kwargs.update({'type': LABEL})
        super().__init__(*args, **kwargs)
