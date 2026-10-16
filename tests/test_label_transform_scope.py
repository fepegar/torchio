"""Image selection for label transforms and their recorded inverses."""

from __future__ import annotations

import copy
from collections.abc import Callable
from functools import partial

import pytest
import torch

import torchio as tio


def _make_subject() -> tio.Subject:
    labels = torch.tensor([0.0, 1.0]).reshape(1, 1, 1, 2)
    return tio.Subject(
        a=tio.LabelMap(labels.clone()),
        b=tio.LabelMap(labels.clone()),
        scan=tio.ScalarImage(labels + 10),
    )


def _make_input(
    subject: tio.Subject, as_batch: bool
) -> tio.Subject | tio.SubjectsBatch:
    if not as_batch:
        return subject
    other = copy.deepcopy(subject)
    for image in other.images.values():
        image.set_data(image.data.flip(-1))
    return tio.SubjectsBatch.from_subjects([subject, other])


@pytest.mark.parametrize("as_batch", [False, True], ids=["subject", "batch"])
@pytest.mark.parametrize(
    ("factory", "expected_values"),
    [
        pytest.param(partial(tio.RemapLabels, {1: 9}), [[0, 9]], id="remap"),
        pytest.param(partial(tio.RemoveLabels, [1]), [[0, 0]], id="remove"),
        pytest.param(
            partial(tio.OneHot, num_classes=2), [[1, 0], [0, 1]], id="one-hot"
        ),
    ],
)
@pytest.mark.parametrize(
    ("selection", "selected"),
    [
        pytest.param({}, {"a", "b"}, id="all"),
        pytest.param({"include": ["a", "scan"]}, {"a"}, id="include"),
        pytest.param({"exclude": ["b"]}, {"a"}, id="exclude"),
        pytest.param({"include": []}, set(), id="empty-include"),
        pytest.param(
            {"include": ["a", "b"], "exclude": ["b"]}, {"a"}, id="include-exclude"
        ),
        pytest.param(
            {"include": ["a"], "exclude": ["a"]}, set(), id="excluded-selection"
        ),
    ],
)
def test_label_transform_respects_image_selection(
    as_batch: bool,
    factory: Callable[..., tio.Transform],
    expected_values: list[list[int]],
    selection: dict[str, list[str]],
    selected: set[str],
) -> None:
    data = _make_input(_make_subject(), as_batch)
    original = {name: image.data.clone() for name, image in data.images.items()}
    expected = torch.tensor(expected_values, dtype=torch.float32).reshape(-1, 1, 1, 2)
    if as_batch:
        expected = torch.stack([expected, expected.flip(-1)])

    result = factory(**selection)(data)

    for name, before in original.items():
        torch.testing.assert_close(data[name].data, before)
        torch.testing.assert_close(
            result[name].data, expected if name in selected else before
        )


@pytest.mark.parametrize("as_batch", [False, True], ids=["subject", "batch"])
@pytest.mark.parametrize("name", ["remap", "one-hot"])
@pytest.mark.parametrize("selection", [{"include": ["a"]}, {"exclude": ["b"]}])
def test_label_inverse_respects_image_selection(
    as_batch: bool,
    name: str,
    selection: dict[str, list[str]],
) -> None:
    subject = _make_subject()
    if name == "remap":
        # Only the inverse mapping could change this excluded label map.
        subject.b.set_data(torch.full_like(subject.b.data, 9))
        transform = tio.RemapLabels({1: 9}, **selection)
    else:
        # An excluded map may already have several channels before encoding a.
        subject.b.set_data(torch.tensor([1.0, 0.0, 0.0, 1.0]).reshape(2, 1, 1, 2))
        transform = tio.OneHot(num_classes=2, **selection)
    data = _make_input(subject, as_batch)
    original = {key: image.data.clone() for key, image in data.images.items()}

    transformed = transform(data)
    forward = {key: image.data.clone() for key, image in transformed.images.items()}
    restored = transformed.apply_inverse_transform()

    for key, before in original.items():
        torch.testing.assert_close(data[key].data, before)
        torch.testing.assert_close(transformed[key].data, forward[key])
        torch.testing.assert_close(restored[key].data, before)
    torch.testing.assert_close(transformed.b.data, original["b"])
