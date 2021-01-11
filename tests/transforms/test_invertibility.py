import warnings

from torchio.transforms.intensity_transform import IntensityTransform
from ..utils import TorchioTestCase


class TestInvertibility(TorchioTestCase):

    def test_all_random_transforms(self):
        transform = self.get_large_composed_transform()
        # Remove RandomLabelsToImage as it will add a new image to the subject
        for t in transform.transforms:
            if t.name == 'RandomLabelsToImage':
                transform.transforms.remove(t)
                break
        # Ignore elastic deformation and gamma warnings during execution
        # Ignore some transforms not invertible
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            transformed = transform(self.sample_subject)
            inverting_transform = transformed.get_inverse_transform()
            transformed_back = inverting_transform(transformed)
        self.assertEqual(
            transformed.t1.shape,
            transformed_back.t1.shape,
        )
        self.assertTensorEqual(
            transformed.label.affine,
            transformed_back.label.affine,
        )

    def test_ignore_intensity(self):
        composed = self.get_large_composed_transform()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            transformed = composed(self.sample_subject)
        inverse_transform = transformed.get_inverse_transform(warn=False)
        for transform in inverse_transform:
            assert not isinstance(transform, IntensityTransform)
