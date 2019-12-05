from pathlib import Path
from pprint import pprint
from resector import RandomResection
from torchio import ImagesDataset
from torchio.transforms import RandomFlip, RandomAffine
from torchvision.transforms import Compose

import pandas as pd


# image_path = '/tmp/subjects/t1_test_mri_post_smooth_sigma_noise_0.75_sigma_mask_1.nii.gz'
# label_path = '/tmp/subjects/t1_259_resection_seg.nii.gz'

# image_paths = (
#     image_path,
#     image_path.replace('.nii', '_2.nii'),
#     image_path.replace('.nii', '_3.nii'),
#     image_path.replace('.nii', '_4.nii'),
# )

# label_paths = (
#     label_path,
#     label_path.replace('.nii', '_2.nii'),
#     label_path.replace('.nii', '_3.nii'),
#     label_path.replace('.nii', '_4.nii'),
# )

# paths_dict = {
#     'image': image_paths,
#     'label': label_paths,
# }


fns = """1395_gray_matter_left_label.nii.gz
1395_gray_matter_right_label.nii.gz
1395_resectable_left_label.nii.gz
1395_resectable_right_label.nii.gz
1395_t1_pre_gif_seg.nii.gz
1395_unbiased_noise.nii.gz
1395_unbiased.nii.gz""".splitlines()

images_dir = Path('/tmp/transform')
# images_dir = Path('/home/fernando/Desktop/resector_test_image')

paths_dict = dict(
    gray_matter_left=[images_dir / fns[0]],
    gray_matter_right=[images_dir / fns[1]],
    resectable_left=[images_dir / fns[2]],
    resectable_right=[images_dir / fns[3]],
    noise=[images_dir / fns[5]],
    image=[images_dir / fns[6]],
)

scales = (0.9, 1.1)
angles = (-10, 10)
volumes_range = (840, 84000)  # percentiles 1 and 100 of episurg
# df = pd.read_csv('/tmp/volumes.csv')
# volumes = df.Volume.values
axes = (0,)

transforms = (
    RandomResection(volumes_range=volumes_range, verbose=True),
    # RandomAffine(scales=scales, angles=angles, isotropic=False, verbose=True),
    # RandomFlip(axes, verbose=True),
)
transform = Compose(transforms)
dataset = ImagesDataset(paths_dict, transform=transform)

# for i in range(len(dataset)):
#     sample = dataset[i]
#     output_paths_dict = dict(
#         image=f'/tmp/test_{i}.nii.gz',
#         label=f'/tmp/test_label_{i}.nii.gz',
#     )
#     dataset.transform_and_save(sample, output_paths_dict, extract_fg=True)
#     print()


import time
import torch
from torch.utils.data import DataLoader
import numpy as np
torch.manual_seed(42)

# for num_workers in (4, 2, 1, 0):
#     print(num_workers, 'workers')
#     start = time.time()
#     loader = DataLoader(dataset, batch_size=4, num_workers=num_workers)
#     batch = next(iter(loader))
#     duration = time.time() - start
#     print(duration, 'seconds')
#     print()

N = 1
for i in range(N):
    start = time.time()
    sample = dataset[0]
    # print(sample['random_scaling'])
    # print(sample['random_rotation'])
    pprint(sample['random_resection'])
    # print(sample['random_flip'])
    output_paths_dict = dict(
        image=f'/tmp/test_{i}.nii.gz',
        label=f'/tmp/test_label_{i}.nii.gz',
    )
    dataset.save_sample(sample, output_paths_dict, extract_fg=True)
    duration = time.time() - start
    print(duration, 'seconds')
    print()
