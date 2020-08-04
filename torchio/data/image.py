import warnings
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import torch
import humanize
import numpy as np
import nibabel as nib
import SimpleITK as sitk

from ..utils import (
    nib_to_sitk,
    get_rotation_and_spacing_from_affine,
    get_stem,
    ensure_4d,
)
from ..torchio import (
    TypeData,
    TypePath,
    TypeTripletInt,
    TypeTripletFloat,
    DATA,
    TYPE,
    AFFINE,
    PATH,
    STEM,
    INTENSITY,
    LABEL,
)
from .io import read_image, write_image


PROTECTED_KEYS = DATA, AFFINE, TYPE, PATH, STEM


class Image(dict):
    r"""TorchIO image.

    For information about medical image orientation, check out `NiBabel docs`_,
    the `3D Slicer wiki`_, `Graham Wideman's website`_, `FSL docs`_ or
    `SimpleITK docs`_.

    Args:
        path: Path to a file that can be read by
            :mod:`SimpleITK` or :mod:`nibabel`, or to a directory containing
            DICOM files. If :py:attr:`tensor` is given, the data in
            :py:attr:`path` will not be read. The data is expected to be 2D or
            3D, and may have multiple channels (see :attr:`num_spatial_dims` and
            :attr:`channels_last`).
        type: Type of image, such as :attr:`torchio.INTENSITY` or
            :attr:`torchio.LABEL`. This will be used by the transforms to
            decide whether to apply an operation, or which interpolation to use
            when resampling. For example, `preprocessing`_ and `augmentation`_
            intensity transforms will only be applied to images with type
            :attr:`torchio.INTENSITY`. Spatial transforms will be applied to
            all types, and nearest neighbor interpolation is always used to
            resample images with type :attr:`torchio.LABEL`.
            The type :attr:`torchio.SAMPLING_MAP` may be used with instances of
            :py:class:`~torchio.data.sampler.weighted.WeightedSampler`.
        tensor: If :py:attr:`path` is not given, :attr:`tensor` must be a 4D
            :py:class:`torch.Tensor` or NumPy array with dimensions
            :math:`(C, D, H, W)`. If it is not 4D, TorchIO will try to guess
            the dimensions meanings. If 2D, the shape will be interpreted as
            :math:`(H, W)`. If 3D, the number of spatial dimensions should be
            determined in :attr:`num_spatial_dims`. If :attr:`num_spatial_dims`
            is not given and the shape is 3 along the first or last dimensions,
            it will be interpreted as a multichannel 2D image. Otherwise, it
            be interpreted as a 3D image with a single channel.
        affine: If :attr:`path` is not given, :attr:`affine` must be a
            :math:`4 \times 4` NumPy array. If ``None``, :attr:`affine` is an
            identity matrix.
        check_nans: If ``True``, issues a warning if NaNs are found
            in the image. If ``False``, images will not be checked for the
            presence of NaNs.
        num_spatial_dims: If ``2`` and the input tensor has 3 dimensions, it
            will be interpreted as a multichannel 2D image. If ``3`` and the
            input has 3 dimensions, it will be interpreted as a
            single-channel 3D volume.
        channels_last: If ``True``, the last dimension of the input will be
            interpreted as the channels. Defaults to ``True`` if :attr:`path` is
            given and ``False`` otherwise.
        **kwargs: Items that will be added to the image dictionary, e.g.
            acquisition parameters.

    Example:
        >>> import torch
        >>> import torchio
        >>> # Loading from a file
        >>> t1_image = torchio.Image('t1.nii.gz', type=torchio.INTENSITY)
        >>> label_image = torchio.Image('t1_seg.nii.gz', type=torchio.LABEL)
        >>> image = torchio.Image(tensor=torch.rand(3, 4, 5))
        >>> image = torchio.Image('safe_image.nrrd', check_nans=False)
        >>> data, affine = image.data, image.affine
        >>> affine.shape
        (4, 4)
        >>> image.data is image[torchio.DATA]
        True
        >>> image.data is image.tensor
        True
        >>> type(image.data)
        torch.Tensor

    TorchIO images are `lazy loaders`_, i.e. the data is only loaded from disk
    when needed.

    Example:
        >>> import torchio
        >>> image = torchio.Image('t1.nii.gz')
        >>> image  # not loaded yet
        Image(path: t1.nii.gz; type: intensity)
        >>> times_two = 2 * image.data  # data is loaded and cached here
        >>> image
        Image(shape: (1, 256, 256, 176); spacing: (1.00, 1.00, 1.00); orientation: PIR+; memory: 44.0 MiB; type: intensity)
        >>> image.save('doubled_image.nii.gz')

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
            path: Optional[TypePath] = None,
            type: str = None,
            tensor: Optional[TypeData] = None,
            affine: Optional[TypeData] = None,
            check_nans: bool = True,
            num_spatial_dims: Optional[int] = None,
            channels_last: Optional[bool] = None,
            **kwargs: Dict[str, Any],
            ):
        self.check_nans = check_nans
        self.num_spatial_dims = num_spatial_dims

        if type is None:
            warnings.warn(
                'Not specifying the image type is deprecated and will be'
                ' mandatory in the future. You can probably use ScalarImage or'
                ' LabelMap instead'
            )
            type = INTENSITY

        if path is None and tensor is None:
            raise ValueError('A value for path or tensor must be given')
        self._loaded = False

        # Number of channels are typically stored in the last dimensions in disk
        # But if a tensor is given, the channels should be in the first dim
        if channels_last is None:
            channels_last = path is not None
        self.channels_last = channels_last

        tensor = self.parse_tensor(tensor)
        affine = self.parse_affine(affine)
        if tensor is not None:
            self[DATA] = tensor
            self[AFFINE] = affine
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
        properties.append(f'type: {self.type}')
        properties = '; '.join(properties)
        string = f'{self.__class__.__name__}({properties})'
        return string

    def __getitem__(self, item):
        if item in (DATA, AFFINE):
            if item not in self:
                self._load()
        return super().__getitem__(item)

    def __array__(self):
        return self[DATA].numpy()

    def __copy__(self):
        kwargs = dict(
            tensor=self.data,
            affine=self.affine,
            type=self.type,
            path=self.path,
            channels_last=False,
        )
        for key, value in self.items():
            if key in PROTECTED_KEYS: continue
            kwargs[key] = value  # should I copy? deepcopy?
        return self.__class__(**kwargs)

    @property
    def data(self):
        return self[DATA]

    @property
    def tensor(self):
        return self.data

    @property
    def affine(self):
        return self[AFFINE]

    @property
    def type(self):
        return self[TYPE]

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return tuple(self.data.shape)

    @property
    def spatial_shape(self) -> TypeTripletInt:
        return self.shape[1:]

    @property
    def orientation(self):
        return nib.aff2axcodes(self.affine)

    @property
    def spacing(self):
        _, spacing = get_rotation_and_spacing_from_affine(self.affine)
        return tuple(spacing)

    @property
    def memory(self):
        return np.prod(self.shape) * 4  # float32, i.e. 4 bytes per voxel

    def get_spacing_string(self):
        strings = [f'{n:.2f}' for n in self.spacing]
        string = f'({", ".join(strings)})'
        return string

    @staticmethod
    def _parse_path(path: TypePath) -> Path:
        if path is None:
            return None
        try:
            path = Path(path).expanduser()
        except TypeError:
            message = f'Conversion to path not possible for variable: {path}'
            raise TypeError(message)
        if not (path.is_file() or path.is_dir()):  # might be a dir with DICOM
            raise FileNotFoundError(f'File not found: {path}')
        return path

    def parse_tensor(self, tensor: TypeData) -> torch.Tensor:
        if tensor is None:
            return None
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        tensor = self.parse_tensor_shape(tensor)
        if self.check_nans and torch.isnan(tensor).any():
            warnings.warn(f'NaNs found in tensor')
        return tensor

    def parse_tensor_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        return ensure_4d(tensor, self.channels_last, self.num_spatial_dims)

    @staticmethod
    def parse_affine(affine: np.ndarray) -> np.ndarray:
        if affine is None:
            return np.eye(4)
        if not isinstance(affine, np.ndarray):
            raise TypeError(f'Affine must be a NumPy array, not {type(affine)}')
        if affine.shape != (4, 4):
            raise ValueError(f'Affine shape must be (4, 4), not {affine.shape}')
        return affine

    def _load(self) -> Tuple[torch.Tensor, np.ndarray]:
        r"""Load the image from disk.

        Returns:
            Tuple containing a 4D tensor of size :math:`(C, D, H, W)` and a 2D
            :math:`4 \times 4` affine matrix to convert voxel indices to world
            coordinates.
        """
        if self._loaded:
            return
        tensor, affine = read_image(self.path)
        tensor = self.parse_tensor_shape(tensor)

        if self.check_nans and torch.isnan(tensor).any():
            warnings.warn(f'NaNs found in file "{self.path}"')
        self[DATA] = tensor
        self[AFFINE] = affine
        self._loaded = True

    def save(self, path, squeeze=True, channels_last=True):
        """Save image to disk.

        Args:
            path: String or instance of :py:class:`pathlib.Path`.
            squeeze: If ``True``, the singleton dimensions will be removed
                before saving.
            channels_last: If ``True``, the channels will be saved in the last
                dimension.
        """
        write_image(
            self[DATA],
            self[AFFINE],
            path,
            squeeze=squeeze,
            channels_last=channels_last,
        )

    def is_2d(self) -> bool:
        return self.shape[-3] == 1

    def numpy(self) -> np.ndarray:
        """Get a NumPy array containing the image data."""
        return np.asarray(self)

    def as_sitk(self) -> sitk.Image:
        """Get the image as an instance of :py:class:`sitk.Image`."""
        return nib_to_sitk(self[DATA], self[AFFINE])

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

    def set_check_nans(self, check_nans: bool):
        self.check_nans = check_nans

    def crop(self, index_ini: TypeTripletInt, index_fin: TypeTripletInt):
        new_origin = nib.affines.apply_affine(self.affine, index_ini)
        new_affine = self.affine.copy()
        new_affine[:3, 3] = new_origin
        i0, j0, k0 = index_ini
        i1, j1, k1 = index_fin
        patch = self.data[:, i0:i1, j0:j1, k0:k1].clone()
        kwargs = dict(
            tensor=patch,
            affine=new_affine,
            type=self.type,
            path=self.path,
            channels_last=False,
        )
        for key, value in self.items():
            if key in PROTECTED_KEYS: continue
            kwargs[key] = value  # should I copy? deepcopy?
        return self.__class__(**kwargs)


class ScalarImage(Image):
    """Alias for :py:class:`~torchio.Image` of type :py:attr:`torchio.INTENSITY`.

    Example:
        >>> import torch
        >>> import torchio
        >>> image = torchio.ScalarImage('t1.nii.gz')  # loading from a file
        >>> image = torchio.ScalarImage(tensor=torch.rand(128, 128, 68))  # from tensor
        >>> data, affine = image.data, image.affine
        >>> affine.shape
        (4, 4)
        >>> image.data is image[torchio.DATA]
        True
        >>> image.data is image.tensor
        True
        >>> type(image.data)
        torch.Tensor

    See :py:class:`~torchio.Image` for more information.

    Raises:
        ValueError: A :py:attr:`type` is used for instantiation.
    """
    def __init__(self, *args, **kwargs):
        if 'type' in kwargs and kwargs['type'] != INTENSITY:
            raise ValueError('Type of ScalarImage is always torchio.INTENSITY')
        kwargs.update({'type': INTENSITY})
        super().__init__(*args, **kwargs)


class LabelMap(Image):
    """Alias for :py:class:`~torchio.Image` of type :py:attr:`torchio.LABEL`.

    Example:
        >>> import torch
        >>> import torchio
        >>> labels = torchio.LabelMap(tensor=torch.rand(128, 128, 68) > 0.5)
        >>> labels = torchio.LabelMap('t1_seg.nii.gz')  # loading from a file

    See :py:class:`~torchio.data.image.Image` for more information.

    Raises:
        ValueError: If a value for :py:attr:`type` is given.
    """
    def __init__(self, *args, **kwargs):
        if 'type' in kwargs and kwargs['type'] != LABEL:
            raise ValueError('Type of LabelMap is always torchio.LABEL')
        kwargs.update({'type': LABEL})
        super().__init__(*args, **kwargs)
