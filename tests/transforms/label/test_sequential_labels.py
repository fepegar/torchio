import pytest

import torchio as tio

from ...utils import TorchioTestCase


@pytest.mark.parametrize(
    'original_labels',
    (
        (0,),
        (0, 1),
        (0, 1, 2),
        (0, 2),
        (0, 4, 8),
        (1,),
        (1, 2),
        (3, 5, 9, 15, 16, 23),  # values from original @efirdc docstring
        (0, 3, 5, 9, 15, 16, 23),
        (2, 8, 9, 10, 15, 20, 100),  # values from original @efirdc test
        (0, 2, 8, 9, 10, 15, 20, 100),
    ),
)
def test_sequential(original_labels):
    remap_labels = tio.SequentialLabels()
    tensor = TorchioTestCase.get_tensor_with_labels(original_labels)
    subject = tio.Subject(label=tio.LabelMap(tensor=tensor))
    transformed = remap_labels(subject)
    for i, label in enumerate(original_labels):
        original_mask = tensor == label
        new_mask = transformed.label.data == i
        TorchioTestCase.assert_tensor_equal(original_mask, new_mask)
    inverted = transformed.apply_inverse_transform()
    TorchioTestCase.assert_tensor_equal(tensor, inverted.label.data)
