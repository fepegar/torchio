import numpy as np
import torch

from ..data import ScalarImage
from ..data import Subject
from ..data import SubjectsDataset
from ..download import download_url
from ..utils import get_torchio_cache_dir


class MedMNIST(SubjectsDataset):
    """3D MedMNIST v2 datasets.

    Datasets from `MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and
    3D Biomedical Image Classification <https://arxiv.org/abs/2110.14795>`_.

    Please check the `MedMNIST website <https://medmnist.com/>`_ for more
    information, inclusing the license.

    Args:
        split: Dataset split. Should be ``'train'``, ``'val'`` or ``'test'``.
    """

    BASE_URL = 'https://zenodo.org/record/5208230/files'
    SPLITS = 'train', 'training', 'val', 'validation', 'test', 'testing'

    def __init__(self, split, **kwargs):
        if split not in self.SPLITS:
            raise ValueError(f'The split must be one of {self.SPLITS}')
        split = 'train' if split == 'training' else split
        split = 'val' if split == 'validation' else split
        split = 'test' if split == 'testing' else split
        url = f'{self.BASE_URL}/{self.filename}?download=1'
        download_root = get_torchio_cache_dir() / 'MedMNIST'
        download_url(
            url,
            download_root,
            filename=self.filename,
        )
        path = download_root / self.filename
        npz_file = np.load(path)
        images = npz_file[f'{split}_images']
        labels = npz_file[f'{split}_labels']
        subjects = []
        for image, label in zip(images, labels):
            image = ScalarImage(tensor=image[np.newaxis])
            subject = Subject(image=image, labels=torch.from_numpy(label))
            subjects.append(subject)
        super().__init__(subjects, **kwargs)

    @property
    def filename(self):
        return f'{self.__class__.__name__.lower()}.npz'


class OrganMNIST3D(MedMNIST):
    __doc__ = MedMNIST.__doc__


class NoduleMNIST3D(MedMNIST):
    __doc__ = MedMNIST.__doc__


class AdrenalMNIST3D(MedMNIST):
    __doc__ = MedMNIST.__doc__


class FractureMNIST3D(MedMNIST):
    __doc__ = MedMNIST.__doc__


class VesselMNIST3D(MedMNIST):
    __doc__ = MedMNIST.__doc__


class SynapseMNIST3D(MedMNIST):
    __doc__ = MedMNIST.__doc__
