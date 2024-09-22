import pytest

from torchio import RandomSpike

from ...utils import TorchioTestCase


class TestRandomSpike(TorchioTestCase):
    """Tests for `RandomSpike`."""

    def test_with_zero_intensity(self):
        transform = RandomSpike(intensity=0)
        transformed = transform(self.sample_subject)
        self.assert_tensor_almost_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_with_zero_spike(self):
        transform = RandomSpike(num_spikes=0)
        transformed = transform(self.sample_subject)
        self.assert_tensor_almost_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_with_spikes(self):
        transform = RandomSpike()
        transformed = transform(self.sample_subject)
        self.assert_tensor_not_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_negative_num_spikes(self):
        with pytest.raises(ValueError):
            RandomSpike(num_spikes=-1)

    def test_num_spikes_range_with_negative_min(self):
        with pytest.raises(ValueError):
            RandomSpike(num_spikes=(-1, 4))

    def test_not_integer_num_spikes(self):
        with pytest.raises(ValueError):
            RandomSpike(num_spikes=(0.7, 4))

    def test_wrong_num_spikes_type(self):
        with pytest.raises(ValueError):
            RandomSpike(num_spikes='wrong')

    def test_wrong_intensity_type(self):
        with pytest.raises(ValueError):
            RandomSpike(intensity='wrong')
