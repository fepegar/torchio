import torch
import torchio as tio
from torchio import LOCATION, DATA, Subject, ScalarImage
from ...utils import TorchioTestCase


class TestAggregator(TorchioTestCase):
    """Tests for `aggregator` module."""

    def aggregate(self, mode, fixture):
        tensor = torch.ones(1, 1, 4, 4)
        IMG = 'img'
        subject = tio.Subject({IMG: tio.ScalarImage(tensor=tensor)})
        patch_size = 1, 3, 3
        patch_overlap = 0, 2, 2
        sampler = tio.data.GridSampler(subject, patch_size, patch_overlap)
        aggregator = tio.data.GridAggregator(sampler, overlap_mode=mode)
        loader = torch.utils.data.DataLoader(sampler, batch_size=3)
        values_dict = {
            (0, 0): 0,
            (0, 1): 2,
            (1, 0): 4,
            (1, 1): 6,
        }
        for batch in loader:
            for location, data in zip(batch[LOCATION], batch[IMG][DATA]):
                coords_2d = tuple(location[1:3].tolist())
                data *= values_dict[coords_2d]
            aggregator.add_batch(batch[IMG][DATA], batch[LOCATION])
        output = aggregator.get_output_tensor()
        self.assertTensorEqual(output, fixture)

    def test_overlap_crop(self):
        fixture = torch.Tensor((
            (0, 0, 2, 2),
            (0, 0, 2, 2),
            (4, 4, 6, 6),
            (4, 4, 6, 6),
        )).reshape(1, 1, 4, 4)
        self.aggregate('crop', fixture)

    def test_overlap_average(self):
        fixture = torch.Tensor((
            (0, 1, 1, 2),
            (2, 3, 3, 4),
            (2, 3, 3, 4),
            (4, 5, 5, 6),
        )).reshape(1, 1, 4, 4)
        self.aggregate('average', fixture)
