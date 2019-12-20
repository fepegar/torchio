from abc import abstractmethod

import torch
import SimpleITK as sitk

from .transform import Transform


class RandomTransform(Transform):
    def __init__(self, seed=None, verbose=False):
        super().__init__(verbose=verbose)
        self.seed = seed

    def __call__(self, sample):
        self.check_seed()
        return super().__call__(sample)

    @staticmethod
    @abstractmethod
    def get_params():
        pass

    @staticmethod
    def nib_to_sitk(array, affine):
        """
        TODO: figure out how to get directions from affine
        so that I don't need this
        """
        import nibabel as nib
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(suffix='.nii') as f:
            nib.Nifti1Image(array, affine).to_filename(f.name)
            image = sitk.ReadImage(f.name)
        return image

    def check_seed(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
