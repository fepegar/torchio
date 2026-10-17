"""Tests for Motion transform."""

from __future__ import annotations

import pytest
import torch

import torchio as tio
from torchio.transforms.intensity.motion import _apply_motion_per_instance


def _make_subject(with_label: bool = True) -> tio.Subject:
    data = torch.rand(1, 10, 10, 10) * 100
    kwargs: dict = {"t1": tio.ScalarImage(data)}
    if with_label:
        seg = torch.zeros(1, 10, 10, 10, dtype=torch.float32)
        seg[0, 2:5, 2:5, 2:5] = 1
        seg[0, 6:9, 6:9, 6:9] = 2
        kwargs["seg"] = tio.LabelMap(seg)
    return tio.Subject(**kwargs)


class TestMotion:
    def test_changes_data(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Motion(degrees=15, translation=10)(subject)
        assert not torch.allclose(result.t1.data, original)

    def test_num_transforms_validation(self) -> None:
        with pytest.raises(ValueError, match="num_transforms"):
            tio.Motion(num_transforms=0)

    def test_leaves_labels_unchanged(self) -> None:
        subject = _make_subject()
        original_seg = subject.seg.data.clone()
        result = tio.Motion()(subject)
        torch.testing.assert_close(result.seg.data, original_seg)

    def test_preserves_shape(self) -> None:
        subject = _make_subject(with_label=False)
        result = tio.Motion()(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_single_transform(self) -> None:
        subject = _make_subject(with_label=False)
        result = tio.Motion(num_transforms=1)(subject)
        assert result.t1.data.shape == subject.t1.data.shape


class TestMotionPerInstance:
    def _batch(self, batch_size: int = 5) -> tio.SubjectsBatch:
        data = torch.rand(1, 12, 12, 12)
        subjects = [
            tio.Subject(t1=tio.ScalarImage(data.clone())) for _ in range(batch_size)
        ]
        return tio.SubjectsBatch.from_subjects(subjects)

    def test_per_instance_differs_across_batch(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        result = tio.Motion(degrees=(5, 15), translation=(5, 15), num_transforms=2)(
            batch
        )
        params = result.applied_transforms[-1].params
        assert "_batched_keys" in params
        assert len(params["transforms"]) == batch.batch_size
        assert not torch.allclose(result.t1.data[0], result.t1.data[1])

    def test_per_instance_false_is_shared(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        transform = tio.Motion(
            degrees=(5, 15),
            translation=(5, 15),
            num_transforms=2,
            per_instance=False,
        )
        result = transform(batch)
        torch.testing.assert_close(result.t1.data[0], result.t1.data[1])

    def test_single_subject_keeps_scalar_params(self) -> None:
        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 12, 12, 12)))
        result = tio.Motion(degrees=15, translation=10)(subject)
        assert "_batched_keys" not in result.applied_transforms[-1].params


class TestMotionDegenerateSegments:
    def test_too_many_transforms_for_axis_raises(self) -> None:
        # num_transforms + 1 segments cannot exceed the chosen spatial axis
        # size; the transform must raise a clear error rather than silently
        # replacing the whole spectrum. Every axis is too small here, so the
        # error is raised regardless of the randomly chosen axis.
        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 2, 2, 2)))
        with pytest.raises(ValueError, match="motion segments"):
            tio.Motion(degrees=5, translation=5, num_transforms=4)(subject)

    def test_too_many_transforms_for_pinned_axis_raises(self) -> None:
        # Pinning the split to a too-small axis must raise even when other
        # axes are large enough.
        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 2, 8, 8)))
        with pytest.raises(ValueError, match="motion segments"):
            tio.Motion(degrees=5, translation=5, num_transforms=4, axes=(0,))(subject)

    def test_inactive_element_axis_is_not_validated(self) -> None:
        # A gated-out element must not make the batch raise: its placeholder
        # axis (0, too small here) has to be ignored when only active
        # elements split along large-enough axes.
        transform = tio.Motion(degrees=5, translation=5, num_transforms=2)
        data = torch.rand(2, 1, 2, 8, 8)
        motion_transforms = [[], transform._sample_transforms()]
        result = _apply_motion_per_instance(data, motion_transforms, [0, 1])
        assert torch.equal(result[0], data[0])
        assert not torch.equal(result[1], data[1])


class TestMotionAxes:
    def test_invalid_axes_raises(self) -> None:
        with pytest.raises(ValueError, match="axes"):
            tio.Motion(axes=(0, 3))

    def test_empty_axes_raises(self) -> None:
        with pytest.raises(ValueError, match="axes"):
            tio.Motion(axes=())

    def test_axis_is_recorded_in_params(self) -> None:
        subject = _make_subject(with_label=False)
        result = tio.Motion(axes=(1,))(subject)
        assert result.applied_transforms[-1].params["axis"] == 1

    def test_split_axis_is_randomized(self) -> None:
        # With all three axes allowed, the sampled split axis should vary
        # across applications rather than always being the first axis.
        torch.manual_seed(0)
        subject = _make_subject(with_label=False)
        transform = tio.Motion(degrees=15, translation=10)
        axes: set[int] = set()
        for _ in range(30):
            axis = transform(subject).applied_transforms[-1].params["axis"]
            axes.add(int(axis))
        assert len(axes) > 1
        assert axes <= {0, 1, 2}

    def test_pinned_axis_is_reproducible(self) -> None:
        # Pinning the axis should make the result independent of the axis RNG
        # draw: two runs with the same rigid-parameter seed must match.
        subject = _make_subject(with_label=False)
        transform = tio.Motion(degrees=15, translation=10, axes=(2,))
        torch.manual_seed(0)
        first = transform(subject).t1.data
        torch.manual_seed(0)
        second = transform(subject).t1.data
        torch.testing.assert_close(first, second)
