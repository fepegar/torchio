import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchio.inference import GridSampler, GridAggregator

def model(arg):
    """Mock PyTorch model"""
    return arg

patch_size = 128, 128, 128
patch_overlap = 4, 4, 4
batch_size = 6
CHANNELS_DIMENSION = 1

# Let's create a dummy volume
input_array = torch.rand((193, 229, 193)).numpy()

# More info about patch-based inference in NiftyNet docs:
# https://niftynet.readthedocs.io/en/dev/window_sizes.html
grid_sampler = GridSampler(input_array, patch_size, patch_overlap)
aggregator = GridAggregator(input_array, patch_overlap)
patch_loader = DataLoader(grid_sampler, batch_size=batch_size)

with torch.no_grad():
    for patches_batch in tqdm(patch_loader):
        input_tensor = patches_batch['image']
        locations = patches_batch['location']
        logits = model(input_tensor)  # some model
        labels = logits.argmax(dim=CHANNELS_DIMENSION, keepdim=True)
        outputs = labels
        aggregator.add_batch(outputs, locations)

output_array = aggregator.output_array
