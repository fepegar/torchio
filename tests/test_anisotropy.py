"""Tests for Anisotropy transform."""

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


class TestAnisotropy:
    def test_changes_data(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Anisotropy(downsampling=3.0)(subject)
        assert not torch.allclose(result.t1.data, original)

    def test_preserves_shape(self) -> None:
        subject = _make_subject(with_label=False)
        result = tio.Anisotropy(downsampling=2.0)(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_respects_exclude(self) -> None:
        torch.manual_seed(0)
        data = torch.rand(1, 10, 10, 10) * 100
        subject = tio.Subject(
            a=tio.ScalarImage(data.clone()), b=tio.ScalarImage(data.clone())
        )
        original_b = subject.b.data.clone()
        result = tio.Anisotropy(downsampling=3.0, exclude=["b"])(subject)
        torch.testing.assert_close(result.b.data, original_b)

    def test_specific_axis(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Anisotropy(axes=(0,), downsampling=3.0)(subject)
        assert not torch.allclose(result.t1.data, original)

    def test_labels_use_nearest(self) -> None:
        subject = _make_subject()
        result = tio.Anisotropy(downsampling=2.0)(subject)
        unique = result.seg.data.unique().tolist()
        for v in unique:
            assert v == int(v)

    def test_factor_one_is_identity(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Anisotropy(downsampling=1.0)(subject)
        torch.testing.assert_close(result.t1.data, original)


class TestAnisotropyPerInstance:
    def _batch(self, batch_size: int = 6) -> tio.SubjectsBatch:
        data = torch.rand(1, 12, 12, 12)
        subjects = [
            tio.Subject(t1=tio.ScalarImage(data.clone())) for _ in range(batch_size)
        ]
        return tio.SubjectsBatch.from_subjects(subjects)

    def test_per_instance_differs_across_batch(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        result = tio.Anisotropy(downsampling=(2.0, 5.0))(batch)
        params = result.applied_transforms[-1].params
        assert "_batched_keys" in params
        assert len(params["factor"]) == batch.batch_size
        assert not torch.allclose(result.t1.data[0], result.t1.data[1])

    def test_per_instance_false_is_shared(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        result = tio.Anisotropy(downsampling=(2.0, 5.0), per_instance=False)(batch)
        torch.testing.assert_close(result.t1.data[0], result.t1.data[1])

    def test_single_subject_keeps_scalar_params(self) -> None:
        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 12, 12, 12)))
        result = tio.Anisotropy(downsampling=(2.0, 5.0))(subject)
        assert "_batched_keys" not in result.applied_transforms[-1].params


class TestAnisotropyAxisValidation:
    def test_out_of_range_axis_raises(self) -> None:
        # An active per-element axis outside {0, 1, 2} must raise, matching the
        # scalar path, rather than silently becoming a no-op.
        from torchio.transforms.spatial.anisotropy import (
            _simulate_anisotropy_per_instance,
        )

        with pytest.raises(ValueError, match="axis must be in"):
            _simulate_anisotropy_per_instance(
                torch.rand(2, 1, 8, 8, 8),
                axes=[0, 3],
                factors=[2.0, 2.0],
                mode="linear",
            )
