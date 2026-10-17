"""Tests for the Flip spatial transform."""

from __future__ import annotations

import pytest
import torch

import torchio as tio

HAS_MPS = torch.backends.mps.is_available()


class TestFlip:
    def test_flip_axis_0(self) -> None:
        tensor = torch.arange(8).reshape(1, 2, 2, 2).float()
        image = tio.ScalarImage(tensor)
        subject = tio.Subject(t1=image)
        result = tio.Flip(axes=0)(subject)
        expected = torch.flip(tensor, [1])
        torch.testing.assert_close(result.t1.data, expected)

    def test_flip_updates_affine(self) -> None:
        """The affine must track the reversed voxel order, like Crop/Pad/Reorient."""
        affine = tio.AffineMatrix.from_spacing(
            (2.0, 2.0, 2.0), origin=(10.0, 20.0, 30.0)
        )
        tensor = torch.arange(64).reshape(1, 4, 4, 4).float()
        image = tio.ScalarImage(tensor, affine=affine)
        subject = tio.Subject(t1=image)

        before_last_voxel = subject.t1.affine.apply(torch.tensor([[3.0, 0.0, 0.0]]))
        result = tio.Flip(axes=0)(subject)

        # The data that was at the last voxel along axis 0 is now at voxel 0,
        # and the affine should say so: the world position is unchanged.
        after_first_voxel = result.t1.affine.apply(torch.tensor([[0.0, 0.0, 0.0]]))
        torch.testing.assert_close(before_last_voxel, after_first_voxel)
        assert result.t1.affine.orientation[0] != subject.t1.affine.orientation[0]

    def test_flip_affine_matches_equivalent_reorient(self) -> None:
        """Flipping the R/L axis is geometrically the same as reorienting to LAS."""
        affine = tio.AffineMatrix.from_spacing(
            (2.0, 2.0, 2.0), origin=(10.0, 20.0, 30.0)
        )
        tensor = torch.arange(64).reshape(1, 4, 4, 4).float()
        subject = tio.Subject(t1=tio.ScalarImage(tensor, affine=affine))

        flipped = tio.Flip(axes=0)(subject)
        reoriented = tio.Reorient(orientation="LAS")(subject)

        torch.testing.assert_close(flipped.t1.data, reoriented.t1.data)
        assert flipped.t1.affine == reoriented.t1.affine

    def test_flip_twice_restores_affine(self) -> None:
        affine = tio.AffineMatrix.from_spacing(
            (2.0, 2.0, 2.0), origin=(10.0, 20.0, 30.0)
        )
        tensor = torch.rand(1, 4, 5, 6)
        subject = tio.Subject(t1=tio.ScalarImage(tensor.clone(), affine=affine))

        flip = tio.Flip(axes=(0, 1, 2))
        result = flip(flip(subject))
        assert result.t1.affine == subject.t1.affine

    def test_flip_single_int_axis(self) -> None:
        """axes=0 should work the same as axes=(0,)."""
        tensor = torch.arange(8).reshape(1, 2, 2, 2).float()
        s1 = tio.Subject(t1=tio.ScalarImage(tensor.clone()))
        s2 = tio.Subject(t1=tio.ScalarImage(tensor.clone()))
        r1 = tio.Flip(axes=0)(s1)
        r2 = tio.Flip(axes=(0,))(s2)
        torch.testing.assert_close(r1.t1.data, r2.t1.data)

    def test_flip_multiple_axes(self) -> None:
        tensor = torch.arange(8).reshape(1, 2, 2, 2).float()
        image = tio.ScalarImage(tensor)
        subject = tio.Subject(t1=image)
        result = tio.Flip(axes=(0, 1))(subject)
        expected = torch.flip(tensor, [1, 2])
        torch.testing.assert_close(result.t1.data, expected)

    def test_flip_all_images(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
            seg=tio.LabelMap(torch.randint(0, 3, (1, 4, 4, 4))),
        )
        original_t1 = subject.t1.data.clone()
        original_seg = subject.seg.data.clone()
        result = tio.Flip(axes=2)(subject)
        assert not torch.equal(result.t1.data, original_t1)
        assert not torch.equal(result.seg.data, original_seg)

    def test_flip_is_self_inverse(self) -> None:
        tensor = torch.rand(1, 4, 5, 6)
        image = tio.ScalarImage(tensor.clone())
        subject = tio.Subject(t1=image)
        flip = tio.Flip(axes=(0, 1, 2))
        result = flip(flip(subject))
        torch.testing.assert_close(result.t1.data, tensor)

    def test_flip_with_probability_zero(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        original = subject.t1.data.clone()
        result = tio.Flip(axes=0, p=0.0)(subject)
        torch.testing.assert_close(result.t1.data, original)

    def test_flip_probability_per_axis(self) -> None:
        """flip_probability=0 should not flip any axis."""
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        original = subject.t1.data.clone()
        result = tio.Flip(axes=(0, 1, 2), flip_probability=0.0)(subject)
        torch.testing.assert_close(result.t1.data, original)

    def test_flip_probability_one(self) -> None:
        """flip_probability=1 should always flip all specified axes."""
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        original = subject.t1.data.clone()
        result = tio.Flip(axes=(0, 1, 2), flip_probability=1.0)(subject)
        expected = torch.flip(original, [1, 2, 3])
        torch.testing.assert_close(result.t1.data, expected)

    def test_flip_history_recorded(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        result = tio.Flip(axes=1)(subject)
        assert len(result.applied_transforms) == 1
        assert result.applied_transforms[0].name == "Flip"

    def test_flip_accepts_image(self) -> None:
        image = tio.ScalarImage(torch.rand(1, 4, 4, 4))
        result = tio.Flip(axes=0)(image)
        assert isinstance(result, tio.Image)

    def test_flip_accepts_tensor(self) -> None:
        tensor = torch.rand(1, 4, 4, 4)
        result = tio.Flip(axes=0)(tensor)
        assert isinstance(result, torch.Tensor)

    def test_flip_in_compose(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        pipeline = tio.Compose([tio.Flip(axes=0), tio.Flip(axes=1)])
        result = pipeline(subject)
        assert len(result.applied_transforms) == 2

    def test_flip_differentiable(self) -> None:
        tensor = torch.rand(1, 4, 4, 4, requires_grad=True)
        result = tio.Flip(axes=0, copy=False)(tensor)
        loss = result.sum()
        loss.backward()
        assert tensor.grad is not None

    @pytest.mark.skipif(not HAS_MPS, reason="No MPS")
    def test_flip_on_mps(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        subject.to("mps")
        result = tio.Flip(axes=0)(subject)
        assert result.t1.device.type == "mps"

    def test_invalid_axis(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        with pytest.raises(ValueError, match="0, 1, or 2"):
            tio.Flip(axes=3)(subject)

    def test_string_axis(self) -> None:
        """Anatomical label 'Left' should resolve to an axis."""
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        original = subject.t1.data.clone()
        # Default affine is RAS, so 'Left'/'Right' = axis 0
        result = tio.Flip(axes="Left")(subject)
        expected = torch.flip(original, [1])
        torch.testing.assert_close(result.t1.data, expected)

    def test_string_axis_lr(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        result = tio.Flip(axes="LR")(subject)
        assert result.t1.shape == (1, 4, 4, 4)

    def test_invalid_string_axis(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        with pytest.raises(ValueError, match="Unknown anatomical"):
            tio.Flip(axes="X")(subject)

    def test_flip_invertible(self) -> None:
        assert tio.Flip(axes=0).invertible

    def test_flip_inverse_round_trip(self) -> None:
        """Subject.apply_inverse_transform round-trips a flip."""
        tensor = torch.rand(1, 4, 5, 6)
        subject = tio.Subject(t1=tio.ScalarImage(tensor.clone()))
        flip = tio.Flip(axes=(0, 1, 2))
        flipped = flip(subject)
        restored = flipped.apply_inverse_transform()
        torch.testing.assert_close(restored.t1.data, tensor)

    def test_inverse_respects_include_scope(self) -> None:
        a = torch.arange(8.0).reshape(1, 2, 2, 2)
        b = torch.arange(100.0, 108.0).reshape(1, 2, 2, 2)
        subject = tio.Subject(
            a=tio.ScalarImage(a.clone()),
            b=tio.ScalarImage(b.clone()),
        )

        transformed = tio.Flip(axes=(0,), include=["a"])(subject)
        restored = transformed.apply_inverse_transform()

        torch.testing.assert_close(restored.a.data, a)
        torch.testing.assert_close(restored.b.data, b)

    def test_compose_inverse(self) -> None:
        """Compose inverse via Subject.apply_inverse_transform."""
        tensor = torch.rand(1, 4, 5, 6)
        subject = tio.Subject(t1=tio.ScalarImage(tensor.clone()))
        pipeline = tio.Compose(
            [
                tio.Flip(axes=0),
                tio.Flip(axes=1),
            ]
        )
        transformed = pipeline(subject)
        assert len(transformed.applied_transforms) == 2
        restored = transformed.apply_inverse_transform()
        torch.testing.assert_close(restored.t1.data, tensor)

    def test_inverse_on_image_via_subject(self) -> None:
        """Inverse works by copying history to a renamed prediction subject."""
        tensor = torch.rand(1, 4, 5, 6)
        subject = tio.Subject(t1=tio.ScalarImage(tensor.clone()))
        flipped = tio.Flip(axes=0)(subject)
        # Create a prediction subject and copy history
        pred_subject = tio.Subject(
            pred=tio.ScalarImage(flipped.t1.data.clone()),
        )
        pred_subject.applied_transforms = flipped.applied_transforms
        restored = pred_subject.apply_inverse_transform()
        torch.testing.assert_close(restored.pred.data, tensor)

    def test_inverse_skips_non_invertible(self) -> None:
        """Non-invertible transforms are skipped with a warning."""
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        pipeline = tio.Compose(
            [
                tio.Flip(axes=0),
                tio.Noise(std=0.1),  # not invertible
            ]
        )
        transformed = pipeline(subject)
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            restored = transformed.apply_inverse_transform()
            assert any("not invertible" in str(x.message) for x in w)
        assert restored.t1.shape == (1, 4, 4, 4)

    def test_ignore_intensity(self) -> None:
        """ignore_intensity=True skips intensity transforms silently."""
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        pipeline = tio.Compose(
            [
                tio.Flip(axes=0),
                tio.Noise(std=0.1),
            ]
        )
        transformed = pipeline(subject)
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            transformed.apply_inverse_transform(ignore_intensity=True)
            # No warning about Noise since intensity is ignored
            assert not any("Noise" in str(x.message) for x in w)


class TestFlipPerInstance:
    def _batch(self, batch_size: int = 6) -> tio.SubjectsBatch:
        data = torch.rand(1, 8, 8, 8)
        subjects = [
            tio.Subject(t1=tio.ScalarImage(data.clone())) for _ in range(batch_size)
        ]
        return tio.SubjectsBatch.from_subjects(subjects)

    def test_per_instance_axes_differ_across_batch(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        result = tio.Flip(axes=(0, 1, 2), flip_probability=0.5)(batch)
        params = result.applied_transforms[-1].params
        assert "_batched_keys" in params
        assert len(params["axes"]) == batch.batch_size
        distinct = {tuple(a) for a in params["axes"]}
        assert len(distinct) > 1

    def test_per_instance_affines_match_each_elements_own_axes(self) -> None:
        """Each batch element's affine must reflect only its own flipped axes."""
        torch.manual_seed(0)
        affine = tio.AffineMatrix.from_spacing((1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 8, 8, 8), affine=affine.clone()),
            )
            for _ in range(6)
        ]
        batch = tio.SubjectsBatch.from_subjects(subjects)
        result = tio.Flip(axes=(0, 1, 2), flip_probability=0.5)(batch)

        axes_per_element = result.applied_transforms[-1].params["axes"]
        for index, axes in enumerate(axes_per_element):
            # Expected origin: shifted by (size - 1) along each of this
            # element's own flipped axes, independent of the rotation code.
            expected_origin = [0.0, 0.0, 0.0]
            for axis in axes:
                expected_origin[axis] = 7.0  # (size - 1) for an 8-voxel axis
            actual_origin = result.images["t1"].affines[index].origin
            torch.testing.assert_close(
                torch.tensor(actual_origin),
                torch.tensor(expected_origin),
            )

    def test_per_instance_false_is_shared(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        result = tio.Flip(axes=(0, 1, 2), flip_probability=0.5, per_instance=False)(
            batch
        )
        torch.testing.assert_close(result.t1.data[0], result.t1.data[1])

    def test_single_subject_keeps_scalar_params(self) -> None:
        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)))
        result = tio.Flip(axes=(0, 1, 2), flip_probability=0.5)(subject)
        assert "_batched_keys" not in result.applied_transforms[-1].params

    def test_per_instance_inverse_round_trip(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        original = batch.t1.data.clone()
        result = tio.Flip(axes=(0, 1, 2), flip_probability=0.5)(batch)
        restored = result.apply_inverse_transform()
        torch.testing.assert_close(restored.t1.data, original)

    def test_per_instance_inverse_after_unbatch(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        original = batch.t1.data.clone()
        result = tio.Flip(axes=(0, 1, 2), flip_probability=0.5)(batch)
        for index, subject in enumerate(result.unbatch()):
            restored = subject.apply_inverse_transform()
            torch.testing.assert_close(restored.t1.data, original[index])


class TestFlipInclude:
    def test_no_selected_images_is_noop(self) -> None:
        # make_params must resolve orientation from the selected images and
        # return a no-op when include matches nothing.
        torch.manual_seed(0)
        subjects = [
            tio.Subject(t1=tio.ScalarImage(torch.rand(1, 8, 8, 8))) for _ in range(3)
        ]
        batch = tio.SubjectsBatch.from_subjects(subjects)
        original = batch.t1.data.clone()
        result = tio.Flip(axes=0, include=["missing"])(batch)
        assert torch.equal(result.t1.data, original)
        assert result.applied_transforms[-1].params["axes"] == ()
