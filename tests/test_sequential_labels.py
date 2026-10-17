"""Tests for SequentialLabels transform."""

from __future__ import annotations

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


class TestSequentialLabels:
    def test_basic_sequential(self) -> None:
        seg = torch.zeros(1, 5, 5, 5, dtype=torch.float32)
        seg[0, 0:2, :, :] = 5
        seg[0, 3:5, :, :] = 10
        subject = tio.Subject(seg=tio.LabelMap(seg))
        result = tio.SequentialLabels()(subject)
        unique = sorted(result.seg.data.unique().tolist())
        assert unique == [0.0, 1.0, 2.0]

    def test_already_sequential(self) -> None:
        subject = _make_subject()
        result = tio.SequentialLabels()(subject)
        torch.testing.assert_close(result.seg.data, subject.seg.data)

    def test_inverse(self) -> None:
        seg = torch.zeros(1, 5, 5, 5, dtype=torch.float32)
        seg[0, 0:2, :, :] = 5
        seg[0, 3:5, :, :] = 10
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 5, 5, 5)),
            seg=tio.LabelMap(seg),
        )
        original = subject.seg.data.clone()
        transformed = tio.SequentialLabels()(subject)
        restored = transformed.apply_inverse_transform()
        torch.testing.assert_close(restored.seg.data, original)

    def test_leaves_scalar_unchanged(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        result = tio.SequentialLabels()(subject)
        torch.testing.assert_close(result.t1.data, original)

    def test_respects_exclude(self) -> None:
        a = torch.zeros(1, 5, 5, 5, dtype=torch.float32)
        a[0, 0:2, :, :] = 5
        a[0, 3:5, :, :] = 10
        b = torch.zeros(1, 5, 5, 5, dtype=torch.float32)
        b[0, 0:2, :, :] = 5  # shares a value with a's remapping domain
        subject = tio.Subject(a=tio.LabelMap(a), b=tio.LabelMap(b))
        original_b = subject.b.data.clone()
        result = tio.SequentialLabels(include=["a"])(subject)
        torch.testing.assert_close(result.b.data, original_b)
