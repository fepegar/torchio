from copy import deepcopy
import numpy as np
import torch
from torchio.transforms import HistogramStandardization
from ...utils import TorchioTestCase


class TestHistogramStandardization(TorchioTestCase):
    """Tests for :py:class:`HistogramStandardization` class."""

    def setUp(self):
        super().setUp()
        self.dataset = self.get_ixi_tiny()

    def test_train_histogram(self):
        samples = [self.dataset[i] for i in range(3)]
        paths = [sample['image']['path'] for sample in samples]
        HistogramStandardization.train(
            paths,
            masking_function=HistogramStandardization.mean,
            output_path=(self.dir / 'landmarks.txt'),
        )
        HistogramStandardization.train(
            paths,
            mask_path=samples[0]['label']['path'],
            output_path=(self.dir / 'landmarks.npy'),
        )

    def test_normalize(self):
        landmarks = np.linspace(0, 100, 13)
        landmarks_dict = {'image': landmarks}
        transform = HistogramStandardization(landmarks_dict)
        transform(self.dataset[0])
