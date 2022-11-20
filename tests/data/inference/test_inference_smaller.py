from torch.utils.data import DataLoader
from torchio import DATA
from torchio import LOCATION
from torchio.data.inference import GridAggregator
from torchio.data.inference import GridSampler

from ...utils import TorchioTestCase


class TestInference(TorchioTestCase):
    """Tests for `inference` module."""

    def test_inference_no_padding(self):
        self.try_inference(None)

    def test_inference_padding(self):
        self.try_inference(3)

    def try_inference(self, padding_mode):
        for mode in ['crop', 'average', 'hann']:
            for n in 17, 27:
                patch_size = 10, 15, n
                model_output_size = 10 - 2, 15 - 2, n - 2
                patch_overlap = 0, 0, 0  # this is important
                batch_size = 6

                grid_sampler = GridSampler(
                    self.sample_subject,
                    patch_size,
                    patch_overlap,
                    padding_mode=padding_mode,
                    model_output_size=model_output_size,
                )
                aggregator = GridAggregator(grid_sampler, overlap_mode=mode)
                patch_loader = DataLoader(grid_sampler, batch_size=batch_size)
                for patches_batch in patch_loader:
                    input_tensor = patches_batch['t1'][DATA]
                    locations = patches_batch[LOCATION]
                    logits = model(input_tensor)  # some model
                    outputs = logits
                    i_ini, j_ini, k_ini = 1, 1, 1
                    i_fin, j_fin, k_fin = (
                        patch_size[0] - 1,
                        patch_size[1] - 1,
                        patch_size[2] - 1,
                    )
                    outputs = outputs[
                        :,
                        :,
                        i_ini:i_fin,
                        j_ini:j_fin,
                        k_ini:k_fin,
                    ]
                    aggregator.add_batch(outputs, locations)

                output = aggregator.get_output_tensor()
                assert (output == -5).all()
                assert output.shape == self.sample_subject.t1.shape


def model(tensor):
    tensor[:] = -5
    return tensor
