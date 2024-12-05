import numpy as np
import pytest
import torch
from packaging.version import Version

from torchio import LabelMap
from torchio import ScalarImage
from torchio import Subject
from torchio import SubjectsDataset
from torchio.transforms import HistogramStandardization

from ...utils import TorchioTestCase


class TestHistogramStandardization(TorchioTestCase):
    """Tests for :class:`HistogramStandardization` class."""

    def setUp(self):
        super().setUp()
        subjects = []
        for i in range(5):
            image = ScalarImage(self.get_image_path(f'hs_image_{i}'))
            label_path = self.get_image_path(
                f'hs_label_{i}',
                binary=True,
                force_binary_foreground=True,
            )
            label = LabelMap(label_path)
            subject = Subject(image=image, label=label)
            subjects.append(subject)
        self.subjects = subjects
        self.dataset = SubjectsDataset(self.subjects)

    def test_train_histogram(self):
        paths = [subject.image.path for subject in self.dataset]
        # Use a function to mask
        HistogramStandardization.train(
            paths,
            masking_function=HistogramStandardization.mean,
            output_path=(self.dir / 'landmarks.txt'),
            progress=False,
        )
        # Use a file to mask
        HistogramStandardization.train(
            paths,
            mask_path=self.dataset[0].label.path,
            output_path=(self.dir / 'landmarks.npy'),
            progress=False,
        )
        # Use files to mask
        masks = [subject.label.path for subject in self.dataset]
        HistogramStandardization.train(
            paths,
            mask_path=masks,
            output_path=(self.dir / 'landmarks_masks.npy'),
            progress=False,
        )

    def test_bad_paths_lengths(self):
        with pytest.raises(ValueError):
            HistogramStandardization.train(
                [1, 2],
                mask_path=[1, 2, 3],
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
        with pytest.raises(KeyError):
            transform(self.dataset[0])

    def test_with_saved_dict(self):
        landmarks = np.linspace(0, 100, 13)
        landmarks_dict = {'image': landmarks}
        landmarks_path = self.dir / 'landmarks_dict.pth'
        torch.save(landmarks_dict, landmarks_path)
        kwargs = {}
        if Version(torch.__version__) >= Version('1.13'):
            kwargs['weights_only'] = False
        landmarks_dict = torch.load(landmarks_path, **kwargs)
        transform = HistogramStandardization(landmarks_dict)
        transform(self.dataset[0])

    def test_with_saved_array(self):
        landmarks = np.linspace(0, 100, 13)
        np.save(self.dir / 'landmarks.npy', landmarks)
        landmarks_dict = {'image': self.dir / 'landmarks.npy'}
        transform = HistogramStandardization(landmarks_dict)
        transform(self.dataset[0])
