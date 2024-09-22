import sys

import pytest
import torch
from parameterized import parameterized

import torchio as tio
from torchio.data import UniformSampler
from torchio.utils import create_dummy_dataset

from ..utils import TorchioTestCase


class TestQueue(TorchioTestCase):
    """Tests for `queue` module."""

    def setUp(self):
        super().setUp()
        self.subjects_list = create_dummy_dataset(
            num_images=10,
            size_range=(10, 20),
            directory=self.dir,
            suffix='.nii',
            force=False,
        )

    def run_queue(self, num_workers=0, **kwargs):
        subjects_dataset = tio.SubjectsDataset(self.subjects_list)
        patch_size = 10
        sampler = UniformSampler(patch_size)
        queue_dataset = tio.Queue(
            subjects_dataset,
            max_length=6,
            samples_per_volume=2,
            sampler=sampler,
            num_workers=num_workers,
            **kwargs,
        )
        _ = str(queue_dataset)
        batch_loader = tio.SubjectsLoader(queue_dataset, batch_size=4)
        for batch in batch_loader:
            _ = batch['one_modality'][tio.DATA]
            _ = batch['segmentation'][tio.DATA]
        return queue_dataset

    def test_queue(self):
        self.run_queue(num_workers=0)

    @pytest.mark.skipif(sys.platform == 'darwin', reason='Takes too long on macOS')
    def test_queue_multiprocessing(self):
        self.run_queue(num_workers=2)

    def test_queue_no_start_background(self):
        self.run_queue(num_workers=0, start_background=False)

    @parameterized.expand([(11,), (12,)])
    def test_different_samples_per_volume(self, max_length):
        image2 = tio.ScalarImage(tensor=2 * torch.ones(1, 1, 1, 1))
        image10 = tio.ScalarImage(tensor=10 * torch.ones(1, 1, 1, 1))
        subject2 = tio.Subject(im=image2, num_samples=2)
        subject10 = tio.Subject(im=image10, num_samples=10)
        dataset = tio.SubjectsDataset([subject2, subject10])
        patch_size = 1
        sampler = UniformSampler(patch_size)
        queue_dataset = tio.Queue(
            dataset,
            max_length=max_length,
            samples_per_volume=3,  # should be ignored
            sampler=sampler,
            shuffle_patches=False,
        )
        batch_loader = tio.SubjectsLoader(queue_dataset, batch_size=6)
        tensors = [batch['im'][tio.DATA] for batch in batch_loader]
        all_numbers = torch.stack(tensors).flatten().tolist()
        assert all_numbers.count(10) == 10
        assert all_numbers.count(2) == 2

    def test_get_memory_string(self):
        queue = self.run_queue()
        memory_string = queue.get_max_memory_pretty()
        assert isinstance(memory_string, str)
