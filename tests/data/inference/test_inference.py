from torch.utils.data import DataLoader
from torchio import LOCATION, DATA
from torchio.data.inference import GridSampler, GridAggregator
from ...utils import TorchioTestCase


class TestInference(TorchioTestCase):
    """Tests for `inference` module."""
    def test_inference_no_padding(self):
        self.try_inference(None)

    def test_inference_padding(self):
        self.try_inference(3)

    def try_inference(self, padding_mode):
        for n in 17, 27:
            patch_size = 10, 15, n
            patch_overlap = 4, 6, 8
            batch_size = 6

            grid_sampler = GridSampler(
                self.sample_subject,
                patch_size,
                patch_overlap,
                padding_mode=padding_mode,
            )
            aggregator = GridAggregator(grid_sampler)
            patch_loader = DataLoader(grid_sampler, batch_size=batch_size)
            for patches_batch in patch_loader:
                input_tensor = patches_batch['t1'][DATA]
                locations = patches_batch[LOCATION]
                logits = model(input_tensor)  # some model
                outputs = logits
                aggregator.add_batch(outputs, locations)

            output = aggregator.get_output_tensor()
            assert (output == -5).all()
            assert output.shape == self.sample_subject.t1.shape


def model(tensor):
    tensor[:] = -5
    return tensor
