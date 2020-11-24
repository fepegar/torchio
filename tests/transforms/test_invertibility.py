import warnings

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
