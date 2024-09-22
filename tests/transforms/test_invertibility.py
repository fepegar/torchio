import copy
import warnings

import torch

import torchio as tio

from ..utils import TorchioTestCase


class TestInvertibility(TorchioTestCase):
    def test_all_random_transforms(self):
        transform = self.get_large_composed_transform()
        # Remove RandomLabelsToImage as it will add a new image to the subject
        for t in transform.transforms:
            if t.name == 'RandomLabelsToImage':
                transform.transforms.remove(t)  # noqa: B038
                break
        # Ignore elastic deformation and gamma warnings during execution
        # Ignore some transforms not invertible
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            transformed = transform(self.sample_subject)
            inverting_transform = transformed.get_inverse_transform()
            transformed_back = inverting_transform(transformed)
        assert transformed.t1.shape == transformed_back.t1.shape
        self.assert_tensor_equal(
            transformed.label.affine,
            transformed_back.label.affine,
        )

    def test_different_interpolation(self):
        def model_probs(subject):
            subject = copy.deepcopy(subject)
            subject.im.set_data(torch.rand_like(subject.im.data))
            return subject

        def model_label(subject):
            subject = model_probs(subject)
            subject.im.set_data(torch.bernoulli(subject.im.data))
            return subject

        transform = tio.RandomAffine(image_interpolation='bspline')
        subject = copy.deepcopy(self.sample_subject)
        tensor = (torch.rand(1, 20, 20, 20) > 0.5).float()  # 0s and 1s
        subject = tio.Subject(im=tio.ScalarImage(tensor=tensor))
        transformed = transform(subject)
        assert transformed.im.data.min() < 0
        assert transformed.im.data.max() > 1

        subject_probs = model_probs(transformed)
        transformed_back = subject_probs.apply_inverse_transform()
        assert transformed_back.im.data.min() < 0
        assert transformed_back.im.data.max() > 1
        transformed_back_linear = subject_probs.apply_inverse_transform(
            image_interpolation='linear',
        )
        assert transformed_back_linear.im.data.min() >= 0
        assert transformed_back_linear.im.data.max() <= 1

        subject_label = model_label(transformed)
        transformed_back = subject_label.apply_inverse_transform()
        assert transformed_back.im.data.min() < 0
        assert transformed_back.im.data.max() > 1
        transformed_back_linear = subject_label.apply_inverse_transform(
            image_interpolation='nearest',
        )
        assert transformed_back_linear.im.data.unique().tolist() == [0, 1]
