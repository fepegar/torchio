import unittest
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchio import IMAGE, LOCATION
from torchio.data.inference import GridSampler, GridAggregator


class TestInference(unittest.TestCase):
    """Tests for `inference` module."""
    def test_inference(self):
        def model(arg):
            """Mock PyTorch model"""
            return arg
        patch_size = 10, 15, 27
        patch_overlap = 4, 5, 8
        batch_size = 6
        CHANNELS_DIMENSION = 1

        # Let's create a dummy volume
        input_array = torch.rand((10, 20, 30)).numpy()
        grid_sampler = GridSampler(input_array, patch_size, patch_overlap)
        patch_loader = DataLoader(grid_sampler, batch_size=batch_size)
        aggregator = GridAggregator(input_array, patch_overlap)

        with torch.no_grad():
            for patches_batch in tqdm(patch_loader):
                input_tensor = patches_batch[IMAGE]
                locations = patches_batch[LOCATION]
                logits = model(input_tensor)  # some model
                labels = logits.argmax(dim=CHANNELS_DIMENSION, keepdim=True)
                outputs = labels
                aggregator.add_batch(outputs, locations)
