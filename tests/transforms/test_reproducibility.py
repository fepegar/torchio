import warnings

from ..utils import TorchioTestCase


class TestReproducibility(TorchioTestCase):
    def test_all_random_transforms(self):
        transform = self.get_large_composed_transform()
        # Ignore elastic deformation and gamma warnings during execution
        with warnings.catch_warnings():  # ignore elastic deformation warning
            warnings.simplefilter('ignore', (RuntimeWarning, UserWarning))
            transformed = transform(self.sample_subject)
        reproducing_transform = transformed.get_composed_history()
        with warnings.catch_warnings():  # ignore elastic deformation warning
            warnings.simplefilter('ignore', (RuntimeWarning, UserWarning))
            new_transformed = reproducing_transform(self.sample_subject)
        self.assert_tensor_equal(transformed.t1.data, new_transformed.t1.data)
        self.assert_tensor_equal(
            transformed.label.data,
            new_transformed.label.data,
        )
