import warnings
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import torch
import numpy as np

from ..torchio import TypePath, DATA, TYPE, AFFINE, PATH, STEM, INTENSITY
from .io import read_image


class Image(dict):
    r"""Class to store information about an image.

    Args:
        path: Path to a file that can be read by
            :mod:`SimpleITK` or :mod:`nibabel` or to a directory containing
            DICOM files.
        type: Type of image, such as :attr:`torchio.INTENSITY` or
            :attr:`torchio.LABEL`. This will be used by the transforms to
            decide whether to apply an operation, or which interpolation to use
            when resampling.
        tensor: If :attr:`path` is not given, :attr:`tensor` must be a 4D
            :py:class:`torch.Tensor` with dimensions :math:`(C, D, H, W)`,
            where :math:`C` is the number of channels and :math:`D, H, W`
            are the spatial dimensions.
        affine: If :attr:`path` is not given, :attr:`affine` must be a
            :math:`4 \times 4` NumPy array. If ``None``, :attr:`affine` is an
            identity matrix.
        **kwargs: Items that will be added to image dictionary within the
            subject sample.
    """

    def __init__(
            self,
            path: Optional[TypePath] = None,
            type: str = INTENSITY,
            tensor: Optional[torch.Tensor] = None,
            affine: Optional[torch.Tensor] = None,
            **kwargs: Dict[str, Any],
            ):
        if path is None and tensor is None:
            raise ValueError('A value for path or tensor must be given')
        if path is not None:
            if tensor is not None or affine is not None:
                message = 'If a path is given, tensor and affine must be None'
                raise ValueError(message)
        self.tensor = self.parse_tensor(tensor)
        self.affine = self.parse_affine(affine)
        if self.affine is None:
            self.affine = np.eye(4)
        for key in (DATA, AFFINE, TYPE, PATH, STEM):
            if key in kwargs:
                raise ValueError(f'Key {key} is reserved. Use a different one')

        super().__init__(**kwargs)
        self.path = self._parse_path(path)
        self.type = type
        self.is_sample = False  # set to True by ImagesDataset

    @staticmethod
    def _parse_path(path: TypePath) -> Path:
        if path is None:
            return
        try:
            path = Path(path).expanduser()
        except TypeError:
            message = f'Conversion to path not possible for variable: {path}'
            raise TypeError(message)
        if not (path.is_file() or path.is_dir()):  # might be a dir with DICOM
            raise FileNotFoundError(f'File not found: {path}')
        return path

    @staticmethod
    def parse_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor is None:
            return None
        num_dimensions = tensor.dim()
        if num_dimensions != 3:
            message = (
                'The input tensor must have 3 dimensions (D, H, W),'
                f' but has {num_dimensions}: {tensor.shape}'
            )
            raise RuntimeError(message)
        tensor = tensor.unsqueeze(0)  # add channels dimension
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

    def load(self, check_nans: bool = True) -> Tuple[torch.Tensor, np.ndarray]:
        r"""Load the image from disk.

        The file is expected to be monomodal/grayscale and 2D or 3D.
        A channels dimension is added to the tensor.

        Args:
            check_nans: If ``True``, issues a warning if NaNs are found
                in the image

        Returns:
            Tuple containing a 4D data tensor of size
            :math:`(1, D_{in}, H_{in}, W_{in})`
            and a 2D 4x4 affine matrix
        """
        if self.path is None:
            return self.tensor, self.affine
        tensor, affine = read_image(self.path)
        # https://github.com/pytorch/pytorch/issues/9410#issuecomment-404968513
        tensor = tensor[(None,) * (3 - tensor.ndim)]  # force to be 3D
        tensor = tensor.unsqueeze(0)  # add channels dimension
        if check_nans and torch.isnan(tensor).any():
            warnings.warn(f'NaNs found in file "{self.path}"')
        return tensor, affine
