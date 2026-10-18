"""Tests for OneHot transform."""

from __future__ import annotations

import pytest
import torch

import torchio as tio


def _make_subject(with_label: bool = True) -> tio.Subject:
    data = torch.rand(1, 10, 10, 10) * 100
    kwargs: dict = {"t1": tio.ScalarImage(data)}
    if with_label:
        seg = torch.zeros(1, 10, 10, 10, dtype=torch.float32)
        seg[0, 2:5, 2:5, 2:5] = 1
        seg[0, 6:9, 6:9, 6:9] = 2
        kwargs["seg"] = tio.LabelMap(seg)
    return tio.Subject(**kwargs)


class TestOneHot:
    def test_one_hot_encoding(self) -> None:
        subject = _make_subject()
        result = tio.OneHot()(subject)
        assert result.seg.data.shape[0] == 3
        assert (result.seg.data.sum(dim=0) == 1).all()

    def test_num_classes(self) -> None:
        subject = _make_subject()
        result = tio.OneHot(num_classes=5)(subject)
        assert result.seg.data.shape[0] == 5

    def test_inverse(self) -> None:
        subject = _make_subject()
        original = subject.seg.data.clone()
        transformed = tio.OneHot()(subject)
        restored = transformed.apply_inverse_transform()
        torch.testing.assert_close(restored.seg.data, original)

    @pytest.mark.parametrize("num_classes", [-1, 1])
    @pytest.mark.parametrize("as_batch", [False, True], ids=["subject", "batch"])
    def test_inverse_background(self, num_classes: int, as_batch: bool) -> None:
        subject = tio.Subject(seg=tio.LabelMap(torch.zeros(1, 3, 4, 5)))
        data = (
            tio.SubjectsBatch.from_subjects([subject, subject]) if as_batch else subject
        )
        original = data.seg.data.clone()

        transformed = tio.OneHot(num_classes=num_classes)(data)
        torch.testing.assert_close(transformed.seg.data, torch.ones_like(original))
        restored = transformed.apply_inverse_transform()

        torch.testing.assert_close(restored.seg.data, original)

    def test_leaves_scalar_images_unchanged(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        result = tio.OneHot()(subject)
        torch.testing.assert_close(result.t1.data, original)
