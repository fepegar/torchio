import torch
import pytest

from torchio.datasets.medmnist import (
    OrganMNIST3D,
    NoduleMNIST3D,
    AdrenalMNIST3D,
    FractureMNIST3D,
    VesselMNIST3D,
    SynapseMNIST3D,
)


classes = (
    OrganMNIST3D,
    NoduleMNIST3D,
    AdrenalMNIST3D,
    FractureMNIST3D,
    VesselMNIST3D,
    SynapseMNIST3D,
)


@pytest.mark.parametrize('class_', classes)
@pytest.mark.parametrize('split', ('train', 'val', 'test'))
def test_load_all(class_, split):
    dataset = class_(split)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
    )
    for _ in loader:
        pass
