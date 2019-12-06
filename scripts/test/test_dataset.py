from pprint import pprint
from torchio import ImagesDataset
from torchio.dataset.images import get_paths_dict_from_data_prameters
from torchio.transforms import RandomFlip, RandomAffine, RandomElasticDeformation
from torchvision.transforms import Compose
import torch

import pandas as pd

data_parameters = {'image': {'csv_file':'/data/romain/data_exemple/file_ms.csv'}, 'label1': {'csv_file':'/data/romain/data_exemple/file_p1.csv'},
            'label2': {'csv_file': '/data/romain/data_exemple/file_p2.csv'}, 'label3': {'csv_file':'/data/romain/data_exemple/file_p3.csv'},
                   'sampler': {'csv_file': '/data/romain/data_exemple/file_mask.csv'}}

paths_dict = get_paths_dict_from_data_prameters(data_parameters)

scales = (0.9, 1.1)
angles = (-10, 10)
axes = (0,)

transforms = (
 #   RandomAffine(scales=scales, angles=angles, isotropic=False, verbose=True),
    RandomFlip(axes, verbose=True),
)
transforms = (RandomElasticDeformation(),)

transform = Compose(transforms)
dataset = ImagesDataset(paths_dict, transform=transform)
dataset_not = ImagesDataset(paths_dict, transform=None)
dataload = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=2)
dataloadnot = torch.utils.data.DataLoader(dataset_not, num_workers=0, batch_size=2)

for dd in dataload:
    break

for ddno in dataloadnot:
    break

from nibabel.viewers import OrthoSlicer3D as ov
import numpy as np

ii = np.squeeze( dd['image'][0,0,:],axis=1)
iio = np.squeeze( ddno['image'][0,0,:],axis=1)

ov(ii)
ov(iio)
