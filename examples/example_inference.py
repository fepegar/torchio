"""
Example of segmenting a big image using patch-based dense inference.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchio import IMAGE, LOCATION
from torchio.data.inference import GridSampler, GridAggregator


patch_size = 128, 128, 128  # or just 128
patch_overlap = 4, 4, 4  # or just 4
batch_size = 6
CHANNELS_DIMENSION = 1

# Let's create a dummy volume
input_array = torch.rand((193, 229, 193)).numpy()

# More info about patch-based inference in NiftyNet docs:
# https://niftynet.readthedocs.io/en/dev/window_sizes.html
grid_sampler = GridSampler(input_array, patch_size, patch_overlap)
patch_loader = DataLoader(grid_sampler, batch_size=batch_size)
aggregator = GridAggregator(input_array, patch_overlap)

model = nn.Identity()  # some Pytorch model

with torch.no_grad():
    for patches_batch in tqdm(patch_loader):
        input_tensor = patches_batch[IMAGE]
        locations = patches_batch[LOCATION]
        logits = model(input_tensor)
        labels = logits.argmax(dim=CHANNELS_DIMENSION, keepdim=True)
        outputs = labels
        aggregator.add_batch(outputs, locations)

output_tensor = aggregator.get_output_tensor()
