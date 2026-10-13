"""Tests for Transpose transform."""

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


class TestTranspose:
    def test_swaps_axes(self) -> None:
        data = torch.rand(1, 8, 10, 12)
        subject = tio.Subject(t1=tio.ScalarImage(data))
        result = tio.Transpose()(subject)
        assert result.t1.data.shape == (1, 12, 10, 8)

    def test_double_transpose_restores_shape(self) -> None:
        data = torch.rand(1, 8, 10, 12)
        subject = tio.Subject(t1=tio.ScalarImage(data))
        result = tio.Transpose()(tio.Transpose()(subject))
        assert result.t1.data.shape == (1, 8, 10, 12)

    def test_inverse(self) -> None:
        data = torch.rand(1, 8, 10, 12)
        subject = tio.Subject(t1=tio.ScalarImage(data))
        original = subject.t1.data.clone()
        transformed = tio.Transpose()(subject)
        restored = transformed.apply_inverse_transform()
        torch.testing.assert_close(restored.t1.data, original)

    def test_is_invertible(self) -> None:
        t = tio.Transpose()
        assert t.invertible is True

    def test_symmetric_shape_unchanged(self) -> None:
        data = torch.rand(1, 10, 10, 10)
        subject = tio.Subject(t1=tio.ScalarImage(data))
        result = tio.Transpose()(subject)
        assert result.t1.data.shape == (1, 10, 10, 10)

    def test_respects_exclude(self) -> None:
        data = torch.rand(1, 8, 10, 12)
        subject = tio.Subject(
            a=tio.ScalarImage(data.clone()), b=tio.ScalarImage(data.clone())
        )
        result = tio.Transpose(exclude=["b"])(subject)
        assert result.a.data.shape == (1, 12, 10, 8)
        assert result.b.data.shape == (1, 8, 10, 12)
