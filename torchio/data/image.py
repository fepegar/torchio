import warnings
from pathlib import Path
from typing import (
    Any,
    Dict,
    Tuple,
)

import torch
import numpy as np

from ..torchio import TypePath, DATA, TYPE, AFFINE, PATH, STEM
from .io import read_image


class Image(dict):
    r"""Class to store information about an image.

    Args:
        path: Path to a file that can be read by
            :mod:`SimpleITK` or :mod:`nibabel` or to a directory containing
            DICOM files.
        type_: Type of image, such as :attr:`torchio.INTENSITY` or
            :attr:`torchio.LABEL`. This will be used by the transforms to
            decide whether to apply an operation, or which interpolation to use
            when resampling.
        **kwargs: Items that will be added to image dictionary within the
            subject sample.
    """

    def __init__(self, path: TypePath, type_: str, **kwargs: Dict[str, Any]):
        for key in (DATA, AFFINE, TYPE, PATH, STEM):
            if key in kwargs:
                raise ValueError(f'Key {key} is reserved. Use a different one')

        super().__init__(**kwargs)
        self.path = self._parse_path(path)
        self.type = type_
        self.is_sample = False  # set to True by ImagesDataset

    @staticmethod
    def _parse_path(path: TypePath) -> Path:
        try:
            path = Path(path).expanduser()
        except TypeError:
            message = f'Conversion to path not possible for variable: {path}'
            raise TypeError(message)
        if not (path.is_file() or path.is_dir()):  # might be a dir with DICOM
            raise FileNotFoundError(f'File not found: {path}')
        return path

    def load(self, check_nans: bool = True) -> Tuple[torch.Tensor, np.ndarray]:
        r"""Load the image from disk.

        The file is expected to be monomodal and 3D. A channels dimension is
        added to the tensor.

        Args:
            check_nans: If ``True``, issues a warning if NaNs are found
                in the image

        Returns:
            Tuple containing a 4D data tensor of size
            :math:`(1, D_{in}, H_{in}, W_{in})`
            and a 2D 4x4 affine matrix
        """
        tensor, affine = read_image(self.path)
        tensor = tensor.unsqueeze(0)  # add channels dimension
        if check_nans and torch.isnan(tensor).any():
            warnings.warn(f'NaNs found in file "{self.path}"')
        return tensor, affine
