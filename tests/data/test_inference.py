import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchio import LOCATION, DATA
from torchio.data.inference import GridSampler, GridAggregator
from ..utils import TorchioTestCase


class TestInference(TorchioTestCase):
    """Tests for `inference` module."""
    def test_inference(self):
        model = nn.Conv3d(1, 1, 3)
        patch_size = 10, 15, 27
        patch_overlap = 4, 5, 8
        batch_size = 6
        CHANNELS_DIMENSION = 1

        grid_sampler = GridSampler(self.sample, patch_size, patch_overlap)
        patch_loader = DataLoader(grid_sampler, batch_size=batch_size)
        aggregator = GridAggregator(self.sample, patch_overlap)

        with torch.no_grad():
            for patches_batch in tqdm(patch_loader):
                input_tensor = patches_batch['t1'][DATA]
                locations = patches_batch[LOCATION]
                logits = model(input_tensor)  # some model
                labels = logits.argmax(dim=CHANNELS_DIMENSION, keepdim=True)
                outputs = labels
                aggregator.add_batch(outputs, locations)

        aggregator.get_output_tensor()
