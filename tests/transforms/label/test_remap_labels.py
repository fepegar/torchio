import pytest

import torchio as tio

from ...utils import TorchioTestCase


@pytest.mark.parametrize(
    'original_label_set',
    (
        {0},
        {0, 1},
        {0, 1, 2},
        {0, 2},
        {1, 2, 5, 6},  # values from original @efirdc test
    ),
)
@pytest.mark.parametrize(
    'remapping',
    (
        {},
        {0: 10},
        {0: 10, 1: 11, 2: 12},
        {0: 1},
        {0: 1, 1: 0},
        {0: 1, 1: 2, 2: 0},
        {2: 1, 5: 1},
        {3: 4},
        {3: 1},
        {1: 2, 2: 1, 5: 10, 6: 11},  # values from original @efirdc test
    ),
)
def test_remap(original_label_set, remapping):
    source_label_set = set(remapping.keys())
    target_label_set = set(remapping.values())
    remap_labels = tio.RemapLabels(remapping=remapping)
    tensor = TorchioTestCase.get_tensor_with_labels(original_label_set)
    subject = tio.Subject(label=tio.LabelMap(tensor=tensor))
    transformed = remap_labels(subject)

    new_label_set = TorchioTestCase.get_unique_labels(transformed.label.data)

    if source_label_set.intersection(original_label_set):
        assert new_label_set.intersection(target_label_set)
    else:
        assert new_label_set == original_label_set

    if len(target_label_set) < len(remapping.keys()):
        with pytest.raises(RuntimeError):
            _ = transformed.apply_inverse_transform()
    else:
        inverse_data = transformed.apply_inverse_transform().label.data
        inverted_label_set = TorchioTestCase.get_unique_labels(inverse_data)
        # Users are warned about this in the docs for the transform
        if target_label_set.isdisjoint(original_label_set):
            assert inverted_label_set == original_label_set
