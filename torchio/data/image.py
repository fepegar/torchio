import warnings
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import torch
import humanize
import numpy as np
import nibabel as nib
import SimpleITK as sitk

from ..utils import nib_to_sitk, get_rotation_and_spacing_from_affine, get_stem
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


class Image(dict):

    PROTECTED_KEYS = DATA, AFFINE, TYPE, PATH, STEM

    r"""TorchIO image.

    TorchIO images are `lazy loaders`_, i.e. the data is only loaded from disk
    when needed.

    Example:
        >>> import torchio
        >>> image = torchio.Image('t1.nii.gz', type=torchio.INTENSITY)
        >>> image  # not loaded yet
        Image(path: t1.nii.gz; type: intensity)
        >>> times_two = 2 * image.data  # data is loaded and cached here
        >>> image
        Image(shape: (1, 256, 256, 176); spacing: (1.00, 1.00, 1.00); orientation: PIR+; memory: 44.0 MiB; type: intensity)
        >>> image.save('doubled_image.nii.gz')

    For information about medical image orientation, check out `NiBabel docs`_,
    the `3D Slicer wiki`_, `Graham Wideman's website`_ or `FSL docs`_.

    Args:
        path: Path to a file that can be read by
            :mod:`SimpleITK` or :mod:`nibabel` or to a directory containing
            DICOM files. If :py:attr:`tensor` is given, the data in
            :py:attr:`path` will not be read.
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
        tensor: If :py:attr:`path` is not given, :attr:`tensor` must be a 3D
            :py:class:`torch.Tensor` or NumPy array with dimensions
            :math:`(D, H, W)`.
        affine: If :attr:`path` is not given, :attr:`affine` must be a
            :math:`4 \times 4` NumPy array. If ``None``, :attr:`affine` is an
            identity matrix.
        check_nans: If ``True``, issues a warning if NaNs are found
            in the image.
        **kwargs: Items that will be added to image dictionary within the
            subject sample.

    Example:
        >>> import torch
        >>> import torchio
        >>> # Loading from a file
        >>> t1_image = torchio.Image('t1.nii.gz', type=torchio.INTENSITY)
        >>> # Also:
        >>> image = torchio.ScalarImage('t1.nii.gz')
        >>> label_image = torchio.Image('t1_seg.nii.gz', type=torchio.LABEL)
        >>> # Also:
        >>> label_image = torchio.LabelMap('t1_seg.nii.gz')
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

    .. _lazy loaders: https://en.wikipedia.org/wiki/Lazy_loading
    .. _preprocessing: https://torchio.readthedocs.io/transforms/preprocessing.html#intensity
    .. _augmentation: https://torchio.readthedocs.io/transforms/augmentation.html#intensity
    .. _NiBabel docs: https://nipy.org/nibabel/image_orientation.html
    .. _3D Slicer wiki: https://www.slicer.org/wiki/Coordinate_systems
    .. _FSL docs: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Orientation%20Explained
    .. _Graham Wideman's website: http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm

    """
    def __init__(
            self,
            path: Optional[TypePath] = None,
            type: str = INTENSITY,
            tensor: Optional[TypeData] = None,
            affine: Optional[TypeData] = None,
            check_nans: bool = True,
            **kwargs: Dict[str, Any],
            ):
        if path is None and tensor is None:
            raise ValueError('A value for path or tensor must be given')
        # if path is not None:
        #     if tensor is not None or affine is not None:
        #         message = 'If a path is given, tensor and affine must be None'
        #         raise ValueError(message)
        self._loaded = False
        tensor = self.parse_tensor(tensor)
        affine = self.parse_affine(affine)
        if tensor is not None:
            if affine is None:
                affine = np.eye(4)
            self[DATA] = tensor
            self[AFFINE] = affine
            self._loaded = True
        for key in self.PROTECTED_KEYS:
            if key in kwargs:
                message = f'Key "{key}" is reserved. Use a different one'
                raise ValueError(message)

        super().__init__(**kwargs)
        self.path = self._parse_path(path)
        self[PATH] = '' if self.path is None else str(self.path)
        self[STEM] = '' if self.path is None else get_stem(self.path)
        self[TYPE] = type
        self.check_nans = check_nans

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
                self.load()
        return super().__getitem__(item)

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

    @staticmethod
    def parse_tensor(tensor: TypeData) -> torch.Tensor:
        if tensor is None:
            return None
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        num_dimensions = tensor.dim()
        if num_dimensions != 3:
            message = (
                'The input tensor must have 3 dimensions (D, H, W),'
                f' but has {num_dimensions}: {tensor.shape}'
            )
            raise RuntimeError(message)
        tensor = tensor.unsqueeze(0)  # add channels dimension
        tensor = tensor.float()
        return tensor

    @staticmethod
    def parse_affine(affine: np.ndarray) -> np.ndarray:
        if affine is None:
            return np.eye(4)
        if not isinstance(affine, np.ndarray):
            raise TypeError(f'Affine must be a NumPy array, not {type(affine)}')
        if affine.shape != (4, 4):
            raise ValueError(f'Affine shape must be (4, 4), not {affine.shape}')
        return affine

    def load(self) -> Tuple[torch.Tensor, np.ndarray]:
        r"""Load the image from disk.

        The file is expected to be monomodal/grayscale and 2D or 3D.
        A channels dimension is added to the tensor.

        Returns:
            Tuple containing a 4D data tensor of size
            :math:`(1, D_{in}, H_{in}, W_{in})`
            and a 2D 4x4 affine matrix
        """
        if self._loaded:
            return
        if self.path is None:
            return
        tensor, affine = read_image(self.path)
        # https://github.com/pytorch/pytorch/issues/9410#issuecomment-404968513
        tensor = tensor[(None,) * (3 - tensor.ndim)]  # force to be 3D
        # Remove next line and uncomment the two following ones once/if this issue
        # gets fixed:
        # https://github.com/pytorch/pytorch/issues/29010
        # See also https://discuss.pytorch.org/t/collating-named-tensors/78650/4
        tensor = tensor.unsqueeze(0)  # add channels dimension
        # name_dimensions(tensor, affine)
        # tensor = tensor.align_to('channels', ...)
        if self.check_nans and torch.isnan(tensor).any():
            warnings.warn(f'NaNs found in file "{self.path}"')
        self[DATA] = tensor
        self[AFFINE] = affine
        self._loaded = True

    def save(self, path):
        """Save image to disk.

        Args:
            path: String or instance of :py:class:`pathlib.Path`.
        """
        tensor = self[DATA].squeeze()  # assume 2D if (1, 1, H, W)
        affine = self[AFFINE]
        write_image(tensor, affine, path)

    def is_2d(self) -> bool:
        return self.shape[-3] == 1

    def numpy(self) -> np.ndarray:
        """Get a NumPy array containing the image data."""
        return self[DATA].numpy()

    def as_sitk(self) -> sitk.Image:
        """Get the image as an instance of :py:class:`sitk.Image`."""
        return nib_to_sitk(self[DATA][0], self[AFFINE])

    def get_center(self, lps: bool = False) -> TypeTripletFloat:
        """Get image center in RAS+ or LPS+ coordinates.

        Args:
            lps: If ``True``, the coordinates will be in LPS+ orientation, i.e.
                the first dimension grows towards the left, etc. Otherwise, the
                coordinates will be in RAS+ orientation.
        """
        image = self.as_sitk()
        size = np.array(image.GetSize())
        center_index = (size - 1) / 2
        l, p, s = image.TransformContinuousIndexToPhysicalPoint(center_index)
        if lps:
            return (l, p, s)
        else:
            return (-l, -p, s)

    def set_check_nans(self, check_nans):
        self.check_nans = check_nans

    def crop(self, index_ini, index_fin):
        new_origin = nib.affines.apply_affine(self.affine, index_ini)
        new_affine = self.affine.copy()
        new_affine[:3, 3] = new_origin
        i0, j0, k0 = index_ini
        i1, j1, k1 = index_fin
        patch = self.data[0, i0:i1, j0:j1, k0:k1].clone()
        kwargs = dict(tensor=patch, affine=new_affine, type=self.type)
        for key, value in self.items():
            if key in (DATA, STEM): continue
            kwargs[key] = value  # should I copy? deepcopy?
        return self.__class__(**kwargs)


class ScalarImage(Image):
    """Alias for :py:class:`~torchio.Image` of type :py:attr:`torchio.INTENSITY`.

    See :py:class:`~torchio.Image` for more information.

    Raises:
        ValueError: A :py:attr:`type` is used for instantiation.
    """
    def __init__(self, *args, **kwargs):
        if 'type' in kwargs:
            raise ValueError('Type of ScalarImage is always torchio.INTENSITY')
        super().__init__(*args, **kwargs, type=INTENSITY)


class LabelMap(Image):
    """Alias for :py:class:`~torchio.Image` of type :py:attr:`torchio.LABEL`.

    See :py:class:`~torchio.Image` for more information.

    Raises:
        ValueError: A :py:attr:`type` is used for instantiation.
    """
    def __init__(self, *args, **kwargs):
        if 'type' in kwargs:
            raise ValueError('Type of LabelMap is always torchio.LABEL')
        super().__init__(*args, **kwargs, type=LABEL)
