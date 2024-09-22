import pytest
import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestMask(TorchioTestCase):
    def test_single_mask(self):
        negated_mask = self.sample_subject.label.data.logical_not()
        masked_voxel_indices = negated_mask.nonzero(as_tuple=True)
        transform = tio.Mask(masking_method='label')
        transformed = transform(self.sample_subject)
        assert (transformed.t1.data[masked_voxel_indices] == 0).all()

    def test_single_mask_nonzero_background(self):
        background_value = 314159
        negated_mask = self.sample_subject.label.data.logical_not()
        masked_voxel_indices = negated_mask.nonzero(as_tuple=True)

        transform = tio.Mask(
            masking_method='label',
            outside_value=background_value,
        )
        transformed = transform(self.sample_subject)

        assert (transformed.t1.data[masked_voxel_indices] == background_value).all()

    def test_mask_specified_label(self):
        mask_label = [1]
        negated_mask = self.sample_subject.label.data.logical_not()
        masked_voxel_indices = negated_mask.nonzero(as_tuple=True)

        transform = tio.Mask(masking_method='label', labels=mask_label)
        transformed = transform(self.sample_subject)

        assert (transformed.t1.data[masked_voxel_indices] == 0).all()

    def test_mask_specified_label_small(self):
        def to_image(*numbers):
            return torch.as_tensor(numbers).reshape(1, 1, 1, len(numbers))

        image_tensor = to_image(1, 6, 7, 3, 0)
        label_tensor = to_image(0, 1, 2, 3, 4)
        mask_labels = [1, 2]
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image_tensor),
            label=tio.LabelMap(tensor=label_tensor),
        )
        transform = tio.Mask(masking_method='label', labels=mask_labels)
        transformed = transform(subject)
        masked_list = transformed.image.data.flatten().tolist()
        assert masked_list == [0, 6, 7, 0, 0]

    def test_mask_example(self):
        subject = self.sample_subject
        negated_mask = subject.label.data.logical_not()
        masked_voxel_indices = negated_mask.nonzero(as_tuple=True)
        transform = tio.Mask(masking_method='label')
        transformed = transform(subject)
        assert (transformed.t1.data[masked_voxel_indices] == 0).all()

    def test_4d(self):
        image = tio.ScalarImage(tensor=torch.rand(3, 4, 5, 6))
        mask = tio.LabelMap(tensor=torch.ones(1, 4, 5, 6))
        subject = tio.Subject(image=image, mask_lm=mask)
        transform = tio.Mask(masking_method='mask_lm')
        with pytest.warns(RuntimeWarning, match='^Expanding.*'):
            masked = transform(subject)
        assert masked.image.shape == image.shape
