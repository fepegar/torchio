import os

import pytest

import torchio as tio
from torchio.datasets.medmnist import AdrenalMNIST3D
from torchio.datasets.medmnist import FractureMNIST3D
from torchio.datasets.medmnist import NoduleMNIST3D
from torchio.datasets.medmnist import OrganMNIST3D
from torchio.datasets.medmnist import SynapseMNIST3D
from torchio.datasets.medmnist import VesselMNIST3D

classes = (
    OrganMNIST3D,
    NoduleMNIST3D,
    AdrenalMNIST3D,
    FractureMNIST3D,
    VesselMNIST3D,
    SynapseMNIST3D,
)


@pytest.mark.slow
@pytest.mark.skipif('CI' in os.environ, reason='Unstable on GitHub Actions')
@pytest.mark.parametrize('class_', classes)
@pytest.mark.parametrize('split', ('train', 'val', 'test'))
def test_load_all(class_, split):
    dataset = class_(split)
    loader = tio.SubjectsLoader(
        dataset,
        batch_size=256,
    )
    for _ in loader:
        pass
