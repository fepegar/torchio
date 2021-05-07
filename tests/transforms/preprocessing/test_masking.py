import torchio as tio
from ...utils import TorchioTestCase


class TestMasking(TorchioTestCase):
    """Tests for :class:`tio.Mask` class."""

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

        transform = tio.Mask(masking_method='label',
                             outside_mask_value=background_value)
        transformed = transform(self.sample_subject)

        assert (transformed.t1.data[masked_voxel_indices]
                == background_value).all()

    def test_mask_specified_label(self):
        mask_label = [1]
        negated_mask = self.sample_subject.label.data.logical_not()
        masked_voxel_indices = negated_mask.nonzero(as_tuple=True)

        transform = tio.Mask(masking_method='label', masking_labels=mask_label)
        transformed = transform(self.sample_subject)

        assert (transformed.t1.data[masked_voxel_indices] == 0).all()

    def test_mask_example(self):
        subject = tio.datasets.Colin27()
        negated_mask = subject.brain.data.logical_not()
        masked_voxel_indices = negated_mask.nonzero(as_tuple=True)

        transform = tio.Mask(masking_method='brain')
        transformed = transform(subject)

        assert (transformed.t1.data[masked_voxel_indices] == 0).all()
