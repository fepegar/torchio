import torch
import numpy as np
from torchio.transforms import HistogramStandardization
from torchio import LabelMap, ScalarImage, Subject, SubjectsDataset
from ...utils import TorchioTestCase


class TestHistogramStandardization(TorchioTestCase):
    """Tests for :class:`HistogramStandardization` class."""

    def setUp(self):
        super().setUp()
        self.subjects = [
            Subject(
                image=ScalarImage(self.get_image_path(f'hs_image_{i}')),
                label=LabelMap(
                    self.get_image_path(
                        f'hs_label_{i}',
                        binary=True,
                        force_binary_foreground=True,
                    ),
                ),
            )
            for i in range(5)
        ]
        self.dataset = SubjectsDataset(self.subjects)

    def test_train_histogram(self):
        paths = [sample['image']['path'] for sample in self.dataset]
        HistogramStandardization.train(
            paths,
            masking_function=HistogramStandardization.mean,
            output_path=(self.dir / 'landmarks.txt'),
        )
        HistogramStandardization.train(
            paths,
            mask_path=self.dataset[0]['label']['path'],
            output_path=(self.dir / 'landmarks.npy'),
        )

    def test_normalize(self):
        landmarks = np.linspace(0, 100, 13)
        landmarks_dict = {'image': landmarks}
        transform = HistogramStandardization(landmarks_dict)
        transform(self.dataset[0])

    def test_wrong_image_key(self):
        landmarks = np.linspace(0, 100, 13)
        landmarks_dict = {'wrong_key': landmarks}
        transform = HistogramStandardization(landmarks_dict)
        with self.assertRaises(KeyError):
            transform(self.dataset[0])

    def test_with_saved_dict(self):
        landmarks = np.linspace(0, 100, 13)
        landmarks_dict = {'image': landmarks}
        torch.save(landmarks_dict, self.dir / 'landmarks_dict.pth')
        landmarks_dict = torch.load(self.dir / 'landmarks_dict.pth')
        transform = HistogramStandardization(landmarks_dict)
        transform(self.dataset[0])

    def test_with_saved_array(self):
        landmarks = np.linspace(0, 100, 13)
        np.save(self.dir / 'landmarks.npy', landmarks)
        landmarks_dict = {'image': self.dir / 'landmarks.npy'}
        transform = HistogramStandardization(landmarks_dict)
        transform(self.dataset[0])
