import torch
import torchio as tio
from einops import rearrange
import matplotlib.pyplot as plt


def plot_medmnist_dataset(class_name):
    rows = 16
    cols = 28

    class_ = getattr(tio.datasets, class_name)
    dataset = class_('train')
    loader = torch.utils.data.DataLoader(dataset, batch_size=rows * cols)
    batch = tio.utils.get_first_item(loader)
    tensor = batch['image'][tio.DATA]
    pattern = '(b1 b2) c x y z -> c x (b1 y) (b2 z)'
    tensor = rearrange(tensor, pattern, b1=rows, b2=cols)
    sx = tensor.shape[1]
    plt.imshow(tensor[0, sx // 2], cmap='gray')
    plt.show()


def plot_organ_mnist():
    plot_medmnist_dataset('OrganMNIST3D')


def plot_nodule_mnist():
    plot_medmnist_dataset('NoduleMNIST3D')


def plot_adrenal_mnist():
    plot_medmnist_dataset('AdrenalMNIST3D')


def plot_fracture_mnist():
    plot_medmnist_dataset('FractureMNIST3D')


def plot_vessel_mnist():
    plot_medmnist_dataset('VesselMNIST3D')


def plot_synapse_mnist():
    plot_medmnist_dataset('SynapseMNIST3D')
