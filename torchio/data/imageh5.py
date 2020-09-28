import numpy as np
import torch
import nibabel as nib

from torchio.data.image import Image
import torchio

class H5DSImage(Image):
    def __init__(self, h5DS=None, lazypatch=True, imtype=torchio.INTENSITY, **kwargs):
        kwargs['path'] = ''
        kwargs['type'] = imtype
        super().__init__(**kwargs)
        self.h5DS = h5DS
        self.lazypatch = lazypatch
        if not self.lazypatch:
            self.load()

    def load(self) -> None:
        if self._loaded:
            return
        if self.lazypatch:
            tensor, affine = self.h5DS, np.eye(4)
        else:
            tensor, affine = self.read_and_check_h5(self.h5DS)
        self[torchio.DATA] = tensor
        self[torchio.AFFINE] = affine
        self._loaded = True

    @property
    def spatial_shape(self):
        if self.lazypatch:
            return self.shape
        else:
            return self.shape[1:]

    def crop(self, index_ini, index_fin):
        new_origin = nib.affines.apply_affine(self.affine, index_ini)
        new_affine = self.affine.copy()
        new_affine[:3, 3] = new_origin
        i0, j0, k0 = index_ini
        i1, j1, k1 = index_fin
        if len(self.data.shape) == 4:
            patch = self.data[:, i0:i1, j0:j1, k0:k1]
        else:
            patch = np.expand_dims(self.data[i0:i1, j0:j1, k0:k1], 0)
        if not isinstance(self.data, torch.Tensor):
            patch = torch.from_numpy(patch)
        kwargs = dict(
            tensor=patch,
            affine=new_affine,
            type=self.type,
            path=self.path,
            h5DS=self.h5DS
        )
        for key, value in self.items():
            if key in torchio.data.image.PROTECTED_KEYS: continue
            kwargs[key] = value  
        return self.__class__(**kwargs)

    def read_and_check_h5(self, h5DS):
        tensor, affine = torch.from_numpy(h5DS[()]).unsqueeze(0), np.eye(4)
        tensor = super().parse_tensor_shape(tensor)
        if self.channels_last:
            tensor = tensor.permute(3, 0, 1, 2)
        if self.check_nans and torch.isnan(tensor).any():
            warnings.warn(f'NaNs found in file "{path}"')
        return tensor, affine