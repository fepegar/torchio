import torch

# Image types
INTENSITY = 'intensity'
LABEL = 'label'
SAMPLING_MAP = 'sampling_map'

# Keys for dataset samples
PATH = 'path'
TYPE = 'type'
STEM = 'stem'
DATA = 'data'
AFFINE = 'affine'
TENSOR = 'tensor'

# For aggregator
IMAGE = 'image'
LOCATION = 'location'

# For special collate function
HISTORY = 'history'

# In PyTorch convention
CHANNELS_DIMENSION = 1

# Code repository
REPO_URL = 'https://github.com/TorchIO-project/torchio/'

# Data repository
DATA_REPO = 'https://github.com/TorchIO-project/torchio-data/raw/main/data/'

# Floating point error
MIN_FLOAT_32 = torch.finfo(torch.float32).eps

# For the queue
NUM_SAMPLES = 'num_samples'
