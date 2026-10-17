"""Tests for Lambda transform."""

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


class TestLambda:
    def test_double(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Lambda(lambda x: 2 * x)(subject)
        torch.testing.assert_close(result.t1.data, 2 * original)

    def test_scalar_only(self) -> None:
        subject = _make_subject()
        original_seg = subject.seg.data.clone()
        result = tio.Lambda(lambda x: x * 0, types_to_apply="scalar")(subject)
        assert result.t1.data.sum() == 0
        torch.testing.assert_close(result.seg.data, original_seg)

    def test_label_only(self) -> None:
        subject = _make_subject()
        original_t1 = subject.t1.data.clone()
        result = tio.Lambda(lambda x: x * 0, types_to_apply="label")(subject)
        assert result.seg.data.sum() == 0
        torch.testing.assert_close(result.t1.data, original_t1)

    def test_not_callable_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            tio.Lambda(42)  # type: ignore[arg-type]

    def test_unknown_types_to_apply_applies_all(self) -> None:
        subject = _make_subject()
        result = tio.Lambda(lambda x: x * 0, types_to_apply="unknown")(subject)
        assert result.t1.data.sum() == 0
        assert result.seg.data.sum() == 0

    def test_respects_exclude(self) -> None:
        subject = _make_subject()
        original_t1 = subject.t1.data.clone()
        result = tio.Lambda(lambda x: x * 0, exclude=["t1"])(subject)
        torch.testing.assert_close(result.t1.data, original_t1)
        assert result.seg.data.sum() == 0
