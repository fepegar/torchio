from torchio import DATA
from torchio.data import LabelSampler
from ...utils import TorchioTestCase


class TestLabelSampler(TorchioTestCase):
    """Tests for `LabelSampler` class."""

    def test_label_sampler(self):
        sampler = LabelSampler(5, 'label')
        for patch in sampler(self.sample, num_patches=10):
            self.assertEqual(patch['label'][DATA][0, 2, 2, 2], 1)
