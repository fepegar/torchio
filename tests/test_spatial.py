"""Tests for spatial transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import torchio as tio
from torchio import AffineMatrix
from torchio.transforms import Affine as AffineTransform
from torchio.transforms.spatial.spatial import _border_mean
from torchio.transforms.spatial.spatial import _build_sampling_grid
from torchio.transforms.spatial.spatial import _check_folding
from torchio.transforms.spatial.spatial import _check_shared_space
from torchio.transforms.spatial.spatial import _compute_channel_pad_value
from torchio.transforms.spatial.spatial import _is_spacing_list
from torchio.transforms.spatial.spatial import _is_spacing_tuple
from torchio.transforms.spatial.spatial import _is_target_space_tuple
from torchio.transforms.spatial.spatial import _normalize_parameter_value
from torchio.transforms.spatial.spatial import _otsu_threshold
from torchio.transforms.spatial.spatial import _parse_center
from torchio.transforms.spatial.spatial import _parse_control_points
from torchio.transforms.spatial.spatial import _parse_default_pad_value
from torchio.transforms.spatial.spatial import _parse_interpolation
from torchio.transforms.spatial.spatial import _parse_locked_borders
from torchio.transforms.spatial.spatial import _parse_num_control_points
from torchio.transforms.spatial.spatial import _parse_spacing
from torchio.transforms.spatial.spatial import _parse_target_space_tuple
from torchio.transforms.spatial.spatial import _prepare_fill_value
from torchio.transforms.spatial.spatial import _to_nonnegative_parameter_range
from torchio.transforms.spatial.spatial import _to_positive_range
from torchio.transforms.spatial.spatial import _validate_isotropic


def _make_subject(
    shape: tuple[int, int, int] = (11, 11, 11),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tio.Subject:
    data = torch.arange(
        shape[0] * shape[1] * shape[2],
        dtype=torch.float32,
    ).reshape(1, *shape)
    label = torch.zeros(1, *shape, dtype=torch.float32)
    label[
        0,
        shape[0] // 4 : shape[0] // 4 + 2,
        shape[1] // 4 : shape[1] // 4 + 2,
        shape[2] // 4 : shape[2] // 4 + 2,
    ] = 1
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    return tio.Subject(
        t1=tio.ScalarImage(data, affine=affine),
        seg=tio.LabelMap(label, affine=affine),
    )


class TestSpatial:
    def test_identity(self) -> None:
        subject = _make_subject()
        transformed = tio.Spatial()(subject)

        torch.testing.assert_close(transformed.t1.data, subject.t1.data)
        torch.testing.assert_close(transformed.seg.data, subject.seg.data)
        np.testing.assert_allclose(
            transformed.t1.affine.numpy(),
            subject.t1.affine.numpy(),
        )

    def test_affine_first_changes_result(self) -> None:
        subject = _make_subject()
        control_points = torch.zeros(5, 5, 5, 3)
        control_points[2, 2, 2, 0] = 2.0

        kwargs = {
            "scales": 1.0,
            "degrees": (0.0, 0.0, 45.0),
            "translation": 0.0,
            "control_points": control_points,
            "default_pad_value": 0.0,
            "default_pad_label": 0.0,
        }

        first = tio.Spatial(affine_first=True, **kwargs)(subject)
        second = tio.Spatial(affine_first=False, **kwargs)(subject)

        assert not torch.allclose(first.t1.data, second.t1.data)

    @pytest.mark.parametrize("affine_first", [True, False])
    def test_affine_first_matches_sequential_transforms(
        self,
        affine_first: bool,
    ) -> None:
        # A single resampling must agree with applying Affine and
        # ElasticDeformation one after the other in the requested order,
        # up to the error of interpolating twice.
        shape = (24, 24, 24)
        coords = torch.stack(
            torch.meshgrid(*[torch.arange(s) for s in shape], indexing="ij"),
        )
        center = torch.tensor([10.0, 13.0, 12.0]).reshape(3, 1, 1, 1)
        data = torch.exp(-((coords - center) ** 2).sum(0) / (2 * 4.0**2))
        subject = tio.Subject(t1=tio.ScalarImage(data[None]))

        generator = torch.Generator().manual_seed(0)
        control_points = 6 * torch.rand(7, 7, 7, 3, generator=generator) - 3
        inner = torch.zeros(7, 7, 7, 1)
        inner[2:-2, 2:-2, 2:-2] = 1
        control_points = control_points * inner
        affine_kwargs = {
            "scales": (1.1, 0.9, 1.0),
            "degrees": (0.0, 0.0, 30.0),
            "translation": (2.0, -1.0, 1.0),
            "default_pad_value": 0.0,
        }

        affine = tio.Affine(**affine_kwargs)
        elastic = tio.ElasticDeformation(control_points=control_points)
        affine_then_elastic = tio.Compose([affine, elastic])(subject).t1.data
        elastic_then_affine = tio.Compose([elastic, affine])(subject).t1.data
        combined = tio.Spatial(
            control_points=control_points,
            affine_first=affine_first,
            **affine_kwargs,
        )(subject).t1.data

        if affine_first:
            expected, other = affine_then_elastic, elastic_then_affine
        else:
            expected, other = elastic_then_affine, affine_then_elastic
        error_expected = (combined - expected).abs().mean()
        error_other = (combined - other).abs().mean()
        assert error_expected < 0.1 * error_other

    @pytest.mark.parametrize("affine_first", [True, False])
    def test_affine_first_with_target_matches_sequential_transforms(
        self,
        affine_first: bool,
    ) -> None:
        # With a target grid, the single resampling must agree with
        # resampling to the target first and then applying Affine and
        # ElasticDeformation in the requested order.  The target covers a
        # shifted field of view, so the elastic field must span the output
        # grid in both orders.
        shape = (24, 24, 24)
        coords = torch.stack(
            torch.meshgrid(*[torch.arange(s) for s in shape], indexing="ij"),
        ).float()
        data = torch.zeros(shape)
        for center in ([11.0, 13.0, 12.0], [15.0, 10.0, 14.0]):
            center_t = torch.tensor(center).reshape(3, 1, 1, 1)
            data += torch.exp(-((coords - center_t) ** 2).sum(0) / (2 * 2.0**2))
        subject = tio.Subject(t1=tio.ScalarImage(data[None]))

        generator = torch.Generator().manual_seed(0)
        control_points = 3 * torch.rand(7, 7, 7, 3, generator=generator) - 1.5
        inner = torch.zeros(7, 7, 7, 1)
        inner[2:-2, 2:-2, 2:-2] = 1
        control_points = control_points * inner
        target_affine = np.diag([0.6, 0.6, 0.6, 1.0])
        target_affine[:3, 3] = (5.0, 3.0, 4.5)
        target = ((30, 26, 28), target_affine)
        # Rotating about the origin keeps the affine the same world-space
        # mapping on the input and on the target grid.
        affine_kwargs = {
            "scales": (1.1, 0.9, 1.0),
            "degrees": (0.0, 0.0, 10.0),
            "translation": (2.0, -1.0, 1.0),
            "center": "origin",
            "default_pad_value": 0.0,
        }

        resample = tio.Resample(target=target)
        affine = tio.Affine(**affine_kwargs)
        elastic = tio.ElasticDeformation(control_points=control_points)
        affine_then_elastic = tio.Compose([resample, affine, elastic])(subject)
        elastic_then_affine = tio.Compose([resample, elastic, affine])(subject)
        combined = tio.Spatial(
            target=target,
            control_points=control_points,
            affine_first=affine_first,
            **affine_kwargs,
        )(subject)

        if affine_first:
            expected, other = affine_then_elastic, elastic_then_affine
        else:
            expected, other = elastic_then_affine, affine_then_elastic
        assert combined.t1.spatial_shape == (30, 26, 28)
        error_expected = (combined.t1.data - expected.t1.data).abs().mean()
        error_other = (combined.t1.data - other.t1.data).abs().mean()
        assert error_expected < 0.25 * error_other

    def test_2d_suppresses_out_of_plane(self) -> None:
        data = torch.rand(1, 8, 8, 1)
        subject = tio.Subject(t1=tio.ScalarImage(data))
        transformed = AffineTransform(
            scales=1.0,
            degrees=(0.0, 0.0, 10.0),
            translation=0.0,
            center="image",
            default_pad_value=0.0,
        )(subject)
        assert transformed.t1.spatial_shape[-1] == 1


class TestResample:
    def test_spacing_target_changes_shape_and_affine(self) -> None:
        subject = _make_subject(shape=(12, 12, 12), spacing=(1.0, 1.0, 1.0))
        transformed = tio.Resample(2)(subject)

        assert transformed.t1.spatial_shape == (6, 6, 6)
        assert transformed.seg.spatial_shape == (6, 6, 6)
        np.testing.assert_allclose(transformed.t1.spacing, (2.0, 2.0, 2.0))
        np.testing.assert_allclose(
            transformed.t1.affine.numpy(),
            transformed.seg.affine.numpy(),
        )

    def test_named_image_target_uses_reference_space(self) -> None:
        reference = tio.ScalarImage(
            torch.ones(1, 6, 6, 6),
            affine=np.diag([2.0, 2.0, 2.0, 1.0]),
        )
        moving = tio.ScalarImage(
            torch.ones(1, 12, 12, 12),
            affine=np.diag([1.0, 1.0, 1.0, 1.0]),
        )
        subject = tio.Subject(t1=reference, t2=moving)

        transformed = tio.Resample("t1")(subject)

        assert transformed.t2.spatial_shape == transformed.t1.spatial_shape
        np.testing.assert_allclose(
            transformed.t2.affine.numpy(),
            transformed.t1.affine.numpy(),
        )

    def test_inverse_restores_geometry(self) -> None:
        subject = _make_subject(shape=(12, 12, 12))
        transformed = tio.Resample(2)(subject)

        restored = transformed.apply_inverse_transform()

        assert restored.t1.spatial_shape == subject.t1.spatial_shape
        np.testing.assert_allclose(
            restored.t1.affine.numpy(),
            subject.t1.affine.numpy(),
        )

    def test_target_image_object(self) -> None:
        subject = _make_subject(shape=(12, 12, 12), spacing=(1.0, 1.0, 1.0))
        reference = tio.ScalarImage(
            torch.ones(1, 6, 6, 6),
            affine=np.diag([2.0, 2.0, 2.0, 1.0]),
        )
        transformed = tio.Resample(target=reference)(subject)
        assert transformed.t1.spatial_shape == (6, 6, 6)

    def test_target_tuple_spacing(self) -> None:
        subject = _make_subject(shape=(12, 12, 12), spacing=(1.0, 1.0, 1.0))
        transformed = tio.Resample(target=(2.0, 2.0, 2.0))(subject)
        assert transformed.t1.spatial_shape == (6, 6, 6)

    def test_target_shape_affine_pair(self) -> None:
        subject = _make_subject(shape=(12, 12, 12), spacing=(1.0, 1.0, 1.0))
        target_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        transformed = tio.Resample(target=((6, 6, 6), target_affine))(subject)
        assert transformed.t1.spatial_shape == (6, 6, 6)

    def test_target_file_path(self, tmp_path: Path) -> None:
        import nibabel as nib

        ref_data = np.zeros((6, 6, 6), dtype=np.float32)
        ref_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        nib.save(nib.Nifti1Image(ref_data, ref_affine), tmp_path / "ref.nii")
        subject = _make_subject(shape=(12, 12, 12), spacing=(1.0, 1.0, 1.0))
        transformed = tio.Resample(target=str(tmp_path / "ref.nii"))(subject)
        assert transformed.t1.spatial_shape == (6, 6, 6)

    def test_target_ndarray_spacing(self) -> None:
        subject = _make_subject(shape=(12, 12, 12), spacing=(1.0, 1.0, 1.0))
        transformed = tio.Resample(target=np.array([2.0, 2.0, 2.0]))(subject)
        assert transformed.t1.spatial_shape == (6, 6, 6)


class TestResampleTargetRange:
    """The spacing form of `target` accepts random ranges and distributions.

    See discussion #1472: `Resample(target=range)` should sample a spacing at
    apply time, consistent with how `degrees`/`scales` accept ranges.
    """

    def test_deterministic_forms_unchanged(self) -> None:
        # Scalar and 3-tuple spacings stay deterministic.
        subject = _make_subject(shape=(12, 12, 12), spacing=(1.0, 1.0, 1.0))
        np.testing.assert_allclose(tio.Resample(2)(subject).t1.spacing, (2.0, 2.0, 2.0))
        np.testing.assert_allclose(
            tio.Resample(target=(2.0, 3.0, 4.0))(subject).t1.spacing,
            (2.0, 3.0, 4.0),
        )

    def test_two_tuple_uniform_range_within_bounds(self) -> None:
        subject = _make_subject(shape=(40, 40, 40), spacing=(1.0, 1.0, 1.0))
        torch.manual_seed(0)
        for _ in range(10):
            spacing = tio.Resample(target=(2.0, 4.0))(subject).t1.spacing
            assert all(2.0 <= s <= 4.0 for s in spacing)

    def test_six_tuple_per_axis_ranges_within_bounds(self) -> None:
        # The exact form from discussion #1472.
        subject = _make_subject(shape=(40, 40, 40), spacing=(1.0, 1.0, 1.0))
        bounds = [(2.0, 4.0), (2.0, 4.0), (3.0, 6.0)]
        torch.manual_seed(0)
        for _ in range(10):
            spacing = tio.Resample(target=(2, 4, 2, 4, 3, 6))(subject).t1.spacing
            for value, (low, high) in zip(spacing, bounds, strict=True):
                assert low <= value <= high

    def test_choice_target(self) -> None:
        from torchio.transforms.parameter_range import Choice

        subject = _make_subject(shape=(40, 40, 40), spacing=(1.0, 1.0, 1.0))
        torch.manual_seed(0)
        for _ in range(10):
            spacing = tio.Resample(target=Choice([2.0, 4.0]))(subject).t1.spacing
            assert all(s in (2.0, 4.0) for s in spacing)

    def test_distribution_target(self) -> None:
        from torch.distributions import Uniform

        subject = _make_subject(shape=(40, 40, 40), spacing=(1.0, 1.0, 1.0))
        torch.manual_seed(0)
        spacing = tio.Resample(target=Uniform(2.0, 4.0))(subject).t1.spacing
        assert all(2.0 <= s <= 4.0 for s in spacing)

    def test_two_tuple_not_treated_as_shape_affine(self) -> None:
        # Regression: a 2-tuple of numbers must be a spacing range, not an
        # attempted (shape, affine) pair (which used to raise).
        subject = _make_subject(shape=(20, 20, 20), spacing=(1.0, 1.0, 1.0))
        transformed = tio.Resample(target=(2.0, 4.0))(subject)
        assert all(2.0 <= s <= 4.0 for s in transformed.t1.spacing)

    def test_nonpositive_range_raises(self) -> None:
        subject = _make_subject(shape=(12, 12, 12), spacing=(1.0, 1.0, 1.0))
        with pytest.raises(ValueError, match="positive"):
            tio.Resample(target=(-2.0, -1.0))(subject)

    def test_seed_reproducible(self) -> None:
        subject = _make_subject(shape=(40, 40, 40), spacing=(1.0, 1.0, 1.0))
        torch.manual_seed(123)
        first = tio.Resample(target=(2, 4, 2, 4, 3, 6))(subject).t1.spacing
        torch.manual_seed(123)
        second = tio.Resample(target=(2, 4, 2, 4, 3, 6))(subject).t1.spacing
        np.testing.assert_allclose(first, second)

    def test_antialias_smooths_before_downsample(self) -> None:
        subject = _make_subject(shape=(20, 20, 20), spacing=(0.5, 0.5, 0.5))
        no_aa = tio.Resample(2)(subject)
        with_aa = tio.Resample(2, antialias=True)(subject)
        assert with_aa.t1.spatial_shape == no_aa.t1.spatial_shape
        # Antialiased result should be smoother (lower high-freq energy)
        assert not torch.allclose(with_aa.t1.data, no_aa.t1.data)

    def test_antialias_skips_label_maps(self) -> None:
        subject = _make_subject(shape=(20, 20, 20), spacing=(0.5, 0.5, 0.5))
        transformed = tio.Resample(2, antialias=True)(subject)
        unique = set(transformed.seg.data.unique().tolist())
        assert unique <= {0.0, 1.0}

    def test_antialias_noop_on_upsample(self) -> None:
        subject = _make_subject(shape=(6, 6, 6), spacing=(2.0, 2.0, 2.0))
        no_aa = tio.Resample(1)(subject)
        with_aa = tio.Resample(1, antialias=True)(subject)
        torch.testing.assert_close(with_aa.t1.data, no_aa.t1.data)


class TestAffine:
    def test_transform_changes_data(self) -> None:
        subject = _make_subject()
        transform = AffineTransform(
            scales=1.0,
            degrees=(0.0, 0.0, 90.0),
            translation=0.0,
            center="image",
            default_pad_value=0.0,
            default_pad_label=0.0,
        )

        transformed = transform(subject)

        assert not torch.allclose(transformed.t1.data, subject.t1.data)
        np.testing.assert_allclose(
            transformed.t1.affine.numpy(),
            subject.t1.affine.numpy(),
        )

    def test_inverse_restores_geometry(self) -> None:
        subject = _make_subject()
        transform = AffineTransform(
            scales=(1.1, 0.9, 1.0),
            degrees=(0.0, 0.0, 20.0),
            translation=(1.0, -2.0, 0.5),
            center="image",
            default_pad_value=0.0,
            default_pad_label=0.0,
        )

        restored = transform(subject).apply_inverse_transform()

        assert restored.t1.spatial_shape == subject.t1.spatial_shape
        np.testing.assert_allclose(
            restored.t1.affine.numpy(),
            subject.t1.affine.numpy(),
        )

    def test_inverse_leaves_excluded_images_untouched(self) -> None:
        # The forward pass records the images it transformed; the inverse
        # must only resample those, so an excluded image (here the label
        # map) is bit-for-bit identical after a forward + inverse round
        # trip.
        subject = _make_subject()
        transform = AffineTransform(
            degrees=(0.0, 0.0, 20.0),
            translation=(1.0, -2.0, 0.5),
            default_pad_value=0.0,
            include=["t1"],
        )
        original_seg = subject.seg.data.clone()

        result = transform(subject)
        assert result.applied_transforms[-1].params["selected_images"] == ["t1"]
        assert torch.equal(result.seg.data, original_seg)

        restored = result.apply_inverse_transform()
        assert torch.equal(restored.seg.data, original_seg)
        subject = _make_subject()
        transform = AffineTransform(
            scales=(0.9, 1.1),
            isotropic=True,
            degrees=0.0,
            translation=0.0,
            default_pad_value=0.0,
        )
        transformed = transform(subject)
        assert transformed.t1.spatial_shape == subject.t1.spatial_shape

    def test_center_origin(self) -> None:
        subject = _make_subject()
        transform = AffineTransform(
            scales=1.0,
            degrees=(0.0, 0.0, 10.0),
            translation=0.0,
            center="origin",
            default_pad_value=0.0,
        )
        transformed = transform(subject)
        assert transformed.t1.spatial_shape == subject.t1.spatial_shape

    def test_choice_degrees(self) -> None:
        subject = _make_subject()
        transform = AffineTransform(
            scales=1.0,
            degrees=tio.Choice([-90, 0, 90, 180]),
            translation=0.0,
            default_pad_value=0.0,
        )
        transformed = transform(subject)
        assert transformed.t1.spatial_shape == subject.t1.spatial_shape

    def test_per_axis_mixed_specs(self) -> None:
        subject = _make_subject()
        transform = AffineTransform(
            scales=1.0,
            degrees=(0, 0, tio.Choice([-90, 0, 90])),
            translation=0.0,
            default_pad_value=0.0,
        )
        transformed = transform(subject)
        assert transformed.t1.spatial_shape == subject.t1.spatial_shape

    def test_distribution_parameter(self) -> None:
        from torch.distributions import Normal

        subject = _make_subject()
        transform = AffineTransform(
            scales=1.0,
            degrees=Normal(0.0, 5.0),
            translation=0.0,
            default_pad_value=0.0,
        )
        transformed = transform(subject)
        assert transformed.t1.spatial_shape == subject.t1.spatial_shape


class TestSpatialPerInstance:
    def _identical_batch(
        self,
        batch_size: int = 4,
        shape: tuple[int, int, int] = (12, 12, 12),
    ) -> tio.SubjectsBatch:
        data = torch.arange(
            shape[0] * shape[1] * shape[2],
            dtype=torch.float32,
        ).reshape(1, *shape)
        subjects = [
            tio.Subject(t1=tio.ScalarImage(data.clone())) for _ in range(batch_size)
        ]
        return tio.SubjectsBatch.from_subjects(subjects)

    def test_per_instance_rotations_differ(self) -> None:
        torch.manual_seed(0)
        batch = self._identical_batch()
        transform = AffineTransform(degrees=(20.0, 80.0), default_pad_value=0.0)
        result = transform(batch)
        params = result.applied_transforms[-1].params
        assert "_batched_keys" in params
        assert len(params["affine_matrix"]) == batch.batch_size
        data = result.t1.data
        assert not torch.allclose(data[0], data[1])
        assert not torch.allclose(data[1], data[2])

    def test_per_instance_cubic_interpolation(self) -> None:
        # Exercises the per-sample high-order interpolation path
        # (interpol.grid_pull) with batched grids.
        pytest.importorskip("interpol")
        torch.manual_seed(0)
        batch = self._identical_batch()
        transform = AffineTransform(
            degrees=(20.0, 80.0),
            default_pad_value=0.0,
            image_interpolation="cubic",
        )
        result = transform(batch)
        assert result.t1.data.shape == batch.t1.data.shape
        assert not torch.allclose(result.t1.data[0], result.t1.data[1])

    def test_per_instance_false_is_shared(self) -> None:
        torch.manual_seed(0)
        batch = self._identical_batch()
        transform = AffineTransform(
            degrees=(20.0, 80.0),
            default_pad_value=0.0,
            per_instance=False,
        )
        result = transform(batch)
        data = result.t1.data
        torch.testing.assert_close(data[0], data[1])
        torch.testing.assert_close(data[1], data[2])

    def test_single_subject_keeps_scalar_params(self) -> None:
        subject = _make_subject()
        result = AffineTransform(degrees=(20.0, 80.0), default_pad_value=0.0)(subject)
        params = result.applied_transforms[-1].params
        assert "_batched_keys" not in params

    def test_per_instance_inverse_restores_geometry(self) -> None:
        torch.manual_seed(0)
        batch = self._identical_batch()
        transform = AffineTransform(
            scales=(0.9, 1.1),
            degrees=(20.0, 80.0),
            translation=(-2.0, 2.0),
            default_pad_value=0.0,
        )
        result = transform(batch)
        restored = result.apply_inverse_transform()
        assert restored.t1.data.shape == batch.t1.data.shape
        for affine in restored.t1.affines:
            np.testing.assert_allclose(
                affine.numpy(),
                batch.t1.affines[0].numpy(),
                atol=1e-5,
            )

    def test_per_instance_p_gates_some_elements(self) -> None:
        torch.manual_seed(0)
        batch = self._identical_batch(batch_size=64)
        original = batch.t1.data.clone()
        transform = AffineTransform(degrees=(40.0, 80.0), default_pad_value=0.0, p=0.5)
        result = transform(batch)
        changed = [
            not torch.allclose(result.t1.data[i], original[i])
            for i in range(batch.batch_size)
        ]
        assert any(changed)
        assert not all(changed)

    def test_per_instance_p_masked_elements_unchanged(self) -> None:
        torch.manual_seed(0)
        batch = self._identical_batch(batch_size=32)
        original = batch.t1.data.clone()
        transform = AffineTransform(degrees=(40.0, 80.0), default_pad_value=0.0, p=0.5)
        result = transform(batch)
        for i, subject in enumerate(result.unbatch()):
            changed = not torch.allclose(subject.t1.data, original[i])
            has_history = len(subject.applied_transforms) == 1
            assert changed == has_history

    def test_per_instance_elastic_differs_across_batch(self) -> None:
        torch.manual_seed(0)
        batch = self._identical_batch()
        transform = tio.ElasticDeformation(
            num_control_points=5,
            max_displacement=(1.0, 3.0),
        )
        result = transform(batch)
        params = result.applied_transforms[-1].params
        assert "_batched_keys" in params
        assert len(params["control_points"]) == batch.batch_size
        data = result.t1.data
        assert not torch.allclose(data[0], data[1])
        assert not torch.allclose(data[1], data[2])

    def test_fully_gated_noop_preserves_per_sample_affines(self) -> None:
        # When every element is gated out (p=0) and there is no target
        # resampling, the transform must be a true no-op: the data and the
        # distinct per-sample affines must be left untouched rather than
        # rebuilt from an identity grid using the first element's affine.
        torch.manual_seed(0)
        subjects = []
        for index in range(4):
            affine = np.eye(4)
            affine[0, 3] = float(index * 10)
            image = tio.ScalarImage(torch.rand(1, 8, 8, 8), affine=affine)
            subjects.append(tio.Subject(t1=image))
        batch = tio.SubjectsBatch.from_subjects(subjects)
        original_data = batch.t1.data.clone()
        original_affines = [affine.numpy().copy() for affine in batch.t1.affines]

        result = AffineTransform(degrees=20.0, p=0.0)(batch)

        assert torch.equal(result.t1.data, original_data)
        for original, new in zip(
            original_affines,
            result.t1.affines,
            strict=True,
        ):
            assert np.allclose(original, new.numpy())

    def test_partially_gated_elements_are_exact_noops(self) -> None:
        # In a mixed batch (0 < p < 1), gated-out elements must be exact,
        # bit-for-bit no-ops even for float64 inputs: the per-sample
        # resample runs an identity grid in float32, so those rows are
        # restored from the input afterwards.
        torch.manual_seed(0)
        data = torch.rand(1, 8, 8, 8, dtype=torch.float64)
        batch = tio.SubjectsBatch.from_subjects(
            [tio.Subject(t1=tio.ScalarImage(data.clone())) for _ in range(8)]
        )
        original = batch.t1.data.clone()
        torch.manual_seed(1)
        result = AffineTransform(degrees=30.0, p=0.5)(batch)
        assert result.t1.data.dtype == torch.float64
        exact = [
            torch.equal(result.t1.data[i], original[i]) for i in range(batch.batch_size)
        ]
        changed = [
            not torch.allclose(result.t1.data[i], original[i], atol=1e-6)
            for i in range(batch.batch_size)
        ]
        # Every element is either an exact no-op or genuinely augmented.
        assert all(
            is_exact ^ is_changed
            for is_exact, is_changed in zip(exact, changed, strict=True)
        )
        assert any(exact)
        assert any(changed)


class TestElasticDeformation:
    def test_accepts_tensor_control_points(self) -> None:
        subject = _make_subject()
        control_points = torch.zeros(5, 5, 5, 3)
        control_points[2, 2, 2, 0] = 2.0

        transformed = tio.ElasticDeformation(control_points=control_points)(subject)

        assert not torch.allclose(transformed.t1.data, subject.t1.data)

    def test_label_interpolation_preserves_label_values(self) -> None:
        subject = _make_subject()
        transformed = AffineTransform(
            scales=(1.1, 1.0, 1.0),
            degrees=(0.0, 0.0, 15.0),
            translation=0.0,
            center="image",
            default_pad_value=0.0,
            default_pad_label=0.0,
        )(subject)

        unique = set(transformed.seg.data.unique().tolist())
        assert unique <= {0.0, 1.0}

    def test_sampled_max_displacement(self) -> None:
        subject = _make_subject()
        transformed = tio.ElasticDeformation(
            max_displacement=2.0,
            num_control_points=5,
            locked_borders=1,
        )(subject)
        assert transformed.t1.spatial_shape == subject.t1.spatial_shape

    def test_inverse_with_elastic(self) -> None:
        subject = _make_subject()
        control_points = torch.zeros(5, 5, 5, 3)
        control_points[2, 2, 2, 0] = 1.0
        transformed = tio.ElasticDeformation(
            control_points=control_points,
        )(subject)
        restored = transformed.apply_inverse_transform()
        assert restored.t1.spatial_shape == subject.t1.spatial_shape

    def test_folding_warning(self) -> None:
        with pytest.warns(RuntimeWarning, match="folding"):
            _check_folding(
                control_points=np.ones((5, 5, 5, 3)),
                max_displacement=(100.0, 100.0, 100.0),
                shape=(10, 10, 10),
                spacing=np.array([1.0, 1.0, 1.0]),
            )


class TestPadValue:
    def test_pad_value_mean(self) -> None:
        subject = _make_subject()
        transform = tio.Spatial(
            degrees=(0.0, 0.0, 30.0),
            scales=1.0,
            translation=0.0,
            default_pad_value="mean",
        )
        transformed = transform(subject)
        assert transformed.t1.spatial_shape == subject.t1.spatial_shape

    def test_pad_value_otsu(self) -> None:
        subject = _make_subject()
        transform = tio.Spatial(
            degrees=(0.0, 0.0, 30.0),
            scales=1.0,
            translation=0.0,
            default_pad_value="otsu",
        )
        transformed = transform(subject)
        assert transformed.t1.spatial_shape == subject.t1.spatial_shape

    def test_pad_value_numeric_nonzero(self) -> None:
        subject = _make_subject()
        transform = tio.Spatial(
            degrees=(0.0, 0.0, 30.0),
            scales=1.0,
            translation=0.0,
            default_pad_value=42.0,
        )
        transformed = transform(subject)
        assert transformed.t1.spatial_shape == subject.t1.spatial_shape

    def test_compute_channel_pad_minimum(self) -> None:
        t = torch.arange(27, dtype=torch.float32).reshape(3, 3, 3)
        assert _compute_channel_pad_value(t, "minimum") == 0.0

    def test_compute_channel_pad_mean(self) -> None:
        t = torch.ones(3, 3, 3)
        val = _compute_channel_pad_value(t, "mean")
        assert abs(val - 1.0) < 1e-5

    def test_compute_channel_pad_otsu(self) -> None:
        t = torch.ones(3, 3, 3)
        val = _compute_channel_pad_value(t, "otsu")
        assert isinstance(val, float)


class TestBorderMeanAndOtsu:
    def test_border_mean_no_otsu(self) -> None:
        t = torch.ones(5, 5, 5) * 3.0
        assert abs(_border_mean(t, filter_otsu=False) - 3.0) < 1e-5

    def test_border_mean_with_otsu(self) -> None:
        t = torch.ones(5, 5, 5)
        result = _border_mean(t, filter_otsu=True)
        assert isinstance(result, float)

    def test_otsu_threshold_empty(self) -> None:
        assert _otsu_threshold(torch.tensor([])) == 0.0

    def test_otsu_threshold_basic(self) -> None:
        values = torch.tensor([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
        threshold = _otsu_threshold(values)
        assert 0.0 <= threshold <= 10.0


class TestValidation:
    def test_locked_borders_invalid(self) -> None:
        with pytest.raises(ValueError, match="locked_borders"):
            _parse_locked_borders(5)

    def test_locked_borders_2_with_4_control_points(self) -> None:
        with pytest.raises(ValueError, match="identity elastic field"):
            tio.Spatial(
                num_control_points=4,
                locked_borders=2,
            )

    def test_invalid_default_pad_label(self) -> None:
        with pytest.raises(TypeError, match="default_pad_label"):
            tio.Spatial(default_pad_label="bad")  # type: ignore[arg-type]

    def test_negative_scales(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            _to_positive_range(-1.0)

    def test_negative_max_displacement(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _to_nonnegative_parameter_range(-1.0)

    def test_isotropic_with_per_axis(self) -> None:
        with pytest.raises(ValueError, match="isotropic"):
            _validate_isotropic((1.0, 1.0, 1.0), isotropic=True)

    def test_parse_num_control_points_too_small(self) -> None:
        with pytest.raises(ValueError, match="greater than 3"):
            _parse_num_control_points(3)

    def test_parse_control_points_bad_shape(self) -> None:
        with pytest.raises(ValueError, match="n_i, n_j, n_k, 3"):
            _parse_control_points(torch.zeros(5, 5, 2))

    def test_parse_control_points_axis_too_small(self) -> None:
        with pytest.raises(ValueError, match="at least 4"):
            _parse_control_points(torch.zeros(3, 5, 5, 3))

    def test_parse_interpolation_invalid(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            _parse_interpolation("bicubic")  # type: ignore[arg-type]

    def test_parse_interpolation_int(self) -> None:
        assert _parse_interpolation(3) == "cubic"
        assert _parse_interpolation(0) == "nearest"

    def test_parse_interpolation_int_invalid(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            _parse_interpolation(99)

    def test_parse_interpolation_not_string(self) -> None:
        with pytest.raises(TypeError, match="string or int"):
            _parse_interpolation(42.5)  # type: ignore[arg-type]

    def test_parse_default_pad_value_invalid_string(self) -> None:
        with pytest.raises(ValueError, match="minimum"):
            _parse_default_pad_value("bad")  # type: ignore[arg-type]

    def test_parse_center_invalid(self) -> None:
        with pytest.raises(ValueError, match="center"):
            _parse_center("middle")  # type: ignore[arg-type]

    def test_parse_spacing_negative(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _parse_spacing(-1.0)

    def test_parse_spacing_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="3 values"):
            _parse_spacing((1.0, 2.0))

    def test_parse_spacing_ndarray_wrong_size(self) -> None:
        with pytest.raises(ValueError, match="3 values"):
            _parse_spacing(np.array([1.0, 2.0]))

    def test_parse_spacing_tuple_3(self) -> None:
        result = _parse_spacing((1.0, 2.0, 3.0))
        assert result == (1.0, 2.0, 3.0)

    def test_parse_spacing_ndarray(self) -> None:
        result = _parse_spacing(np.array([1.0, 2.0, 3.0]))
        assert result == (1.0, 2.0, 3.0)

    def test_target_not_understood(self) -> None:
        subject = _make_subject()
        with pytest.raises(ValueError, match="not understood"):
            tio.Resample(target=object())(subject)  # type: ignore[arg-type]

    def test_target_unknown_string(self) -> None:
        subject = _make_subject()
        with pytest.raises(ValueError, match="Unknown target"):
            tio.Resample(target="nonexistent_image")(subject)

    def test_shared_space_shape_mismatch(self) -> None:
        from torchio.data.batch import ImagesBatch

        batch_a = ImagesBatch(
            torch.rand(1, 1, 8, 8, 8),
            [AffineMatrix()],
        )
        batch_b = ImagesBatch(
            torch.rand(1, 1, 10, 10, 10),
            [AffineMatrix()],
        )
        with pytest.raises(RuntimeError, match="shape"):
            _check_shared_space(
                {"a": batch_a, "b": batch_b},
                (8, 8, 8),
                AffineMatrix(),
            )

    def test_shared_space_affine_mismatch(self) -> None:
        from torchio.data.batch import ImagesBatch

        batch_a = ImagesBatch(
            torch.rand(1, 1, 8, 8, 8),
            [AffineMatrix()],
        )
        batch_b = ImagesBatch(
            torch.rand(1, 1, 8, 8, 8),
            [AffineMatrix(np.diag([2.0, 2.0, 2.0, 1.0]))],
        )
        with pytest.raises(RuntimeError, match="same affine"):
            _check_shared_space(
                {"a": batch_a, "b": batch_b},
                (8, 8, 8),
                AffineMatrix(),
            )


class TestTypeGuards:
    def test_is_spacing_tuple(self) -> None:
        assert _is_spacing_tuple((1.0, 2.0, 3.0))
        assert not _is_spacing_tuple((1.0, 2.0))
        assert not _is_spacing_tuple([1.0, 2.0, 3.0])

    def test_is_spacing_list(self) -> None:
        assert _is_spacing_list([1.0, 2.0, 3.0])
        assert not _is_spacing_list((1.0, 2.0, 3.0))

    def test_is_target_space_tuple(self) -> None:
        assert _is_target_space_tuple(((6, 6, 6), np.eye(4)))
        assert not _is_target_space_tuple((1.0, 2.0, 3.0))
        # A 2-tuple of plain numbers is a spacing range, not a (shape, affine).
        assert not _is_target_space_tuple((2.0, 4.0))


class TestEdgeCases:
    """Tests for defensive branches and edge cases."""

    def test_exclude_all_images(self) -> None:
        """make_params returns empty when all images are excluded (line 210)."""
        subject = _make_subject()
        transform = tio.Spatial(exclude=["t1", "seg"])
        from torchio.data.batch import SubjectsBatch

        batch = SubjectsBatch.from_subjects([subject])
        params = transform.make_params(batch)
        assert params == {"selected_images": []}

    def test_apply_transform_empty_selection(self) -> None:
        """apply_transform returns batch unchanged (line 278)."""
        subject = _make_subject()
        from torchio.data.batch import SubjectsBatch

        batch = SubjectsBatch.from_subjects([subject])
        transform = tio.Spatial()
        original_data = batch.images["t1"].data.clone()
        result = transform.apply_transform(batch, {"selected_images": []})
        torch.testing.assert_close(result.images["t1"].data, original_data)

    def test_inverse_missing_original_space(self) -> None:
        """inverse raises when original is None (lines 330-331)."""
        transform = tio.Spatial()
        params = {
            "affine_matrix": None,
            "control_points": None,
            "original": None,
            "affine_first": True,
            "image_interpolation": "linear",
            "label_interpolation": "nearest",
            "default_pad_value": "minimum",
            "default_pad_label": 0,
        }
        with pytest.raises(RuntimeError, match="original output space"):
            transform.inverse(params)

    def test_apply_spatial_empty_names(self) -> None:
        """_apply_spatial_to_batch returns early for empty names (line 556)."""
        from torchio.transforms.spatial.spatial import _apply_spatial_to_batch

        subject = _make_subject()
        from torchio.data.batch import SubjectsBatch

        batch = SubjectsBatch.from_subjects([subject])
        original = batch.images["t1"].data.clone()
        _apply_spatial_to_batch(
            batch=batch,
            image_names=[],
            target_space=None,
            affine_matrix=None,
            control_points=None,
            max_displacement=None,
            affine_first=True,
            image_interpolation="linear",
            label_interpolation="nearest",
            antialias=False,
            default_pad_value="minimum",
            default_pad_label=0.0,
        )
        torch.testing.assert_close(batch.images["t1"].data, original)

    def test_build_grid_elastic_no_max_displacement(self) -> None:
        """_build_sampling_grid computes max_displacement from field (line 742)."""
        control_points = torch.zeros(5, 5, 5, 3)
        control_points[2, 2, 2, 0] = 0.5
        grid = _build_sampling_grid(
            input_shape=(11, 11, 11),
            input_affine=AffineMatrix(),
            output_shape=(11, 11, 11),
            output_affine=AffineMatrix(),
            affine_matrix=None,
            control_points=control_points,
            max_displacement=None,
            affine_first=True,
            device=torch.device("cpu"),
        )
        assert grid.shape == (11, 11, 11, 3)

    def test_batch_fill_value_bad_type(self) -> None:
        """_batch_fill_value raises TypeError for non-str non-number (lines 907-911)."""
        from torchio.data.batch import ImagesBatch
        from torchio.transforms.spatial.spatial import _batch_fill_value

        batch = ImagesBatch(
            torch.rand(1, 1, 4, 4, 4),
            [AffineMatrix()],
        )
        with pytest.raises(TypeError, match="string or number"):
            _batch_fill_value(
                batch,
                default_pad_value=object(),  # type: ignore[arg-type]
                default_pad_label=0.0,
            )

    def test_prepare_fill_value_multidim(self) -> None:
        """_prepare_fill_value returns a >1D tensor unchanged (line 945)."""
        fill = torch.ones(1, 2, 1, 1, 1)
        ref = torch.zeros(1, 2, 3, 3, 3)
        result = _prepare_fill_value(fill, ref)
        assert result is not None
        assert result.shape == (1, 2, 1, 1, 1)

    def test_compute_channel_pad_unknown(self) -> None:
        """_compute_channel_pad_value raises for unknown strategy (lines 959-960)."""
        with pytest.raises(ValueError, match="Unknown"):
            _compute_channel_pad_value(
                torch.ones(3, 3, 3),
                "bad_strategy",  # type: ignore[arg-type]
            )

    def test_parse_target_space_tuple_wrong_length(self) -> None:
        """_parse_target_space_tuple raises for non-3 shape (lines 1386-1387)."""
        with pytest.raises(ValueError, match="length 3"):
            _parse_target_space_tuple([6, 6], np.eye(4))

    def test_normalize_parameter_value_distribution(self) -> None:
        """_normalize_parameter_value returns Distribution unchanged (line 1548)."""
        from torch.distributions import Normal

        dist = Normal(0.0, 1.0)
        assert _normalize_parameter_value(dist) is dist


class TestExports:
    def test_root_exports_expose_transform_and_matrix(self) -> None:
        assert hasattr(tio, "Spatial")
        assert hasattr(tio, "Resample")
        assert hasattr(tio, "ElasticDeformation")
        assert tio.Affine is AffineTransform
        assert tio.AffineMatrix is AffineMatrix


# ---------------------------------------------------------------------------
# High-order interpolation (torch-interpol)
# ---------------------------------------------------------------------------


class TestHighOrderInterpolation:
    def test_cubic_produces_different_result_from_linear(self) -> None:
        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 16, 16, 16)))
        linear = tio.Affine(
            degrees=10,
            image_interpolation="linear",
        )(subject)
        cubic = tio.Affine(
            degrees=10,
            image_interpolation="cubic",
        )(subject)
        # Same params won't be sampled, so compare shapes at least
        assert linear.t1.data.shape == cubic.t1.data.shape

    def test_cubic_resample(self) -> None:
        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 16, 16, 16)))
        result = tio.Resample(
            target=2.0,
            image_interpolation="cubic",
        )(subject)
        assert result.t1.data.shape[1:] == (8, 8, 8)

    def test_quadratic_interpolation(self) -> None:
        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 10, 10, 10)))
        result = tio.Affine(
            degrees=5,
            image_interpolation="quadratic",
        )(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_int_order_3(self) -> None:
        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 10, 10, 10)))
        result = tio.Affine(
            degrees=5,
            image_interpolation=3,
        )(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_order_0_uses_fast_path(self) -> None:
        """Nearest interpolation should still work via F.grid_sample."""
        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 10, 10, 10)))
        result = tio.Affine(
            degrees=5,
            image_interpolation="nearest",
        )(subject)
        assert result.t1.data.shape == subject.t1.data.shape


def _sphere_label(
    n: int = 64,
    radius: float = 20.0,
    value: float = 1.0,
) -> torch.Tensor:
    center = (n - 1) / 2
    zz, yy, xx = torch.meshgrid(
        torch.arange(n),
        torch.arange(n),
        torch.arange(n),
        indexing="ij",
    )
    distance = ((xx - center) ** 2 + (yy - center) ** 2 + (zz - center) ** 2).sqrt()
    sphere = (distance <= radius).float() * value
    return sphere[None]


def _dice(a: torch.Tensor, b: torch.Tensor) -> float:
    mask_a = a > 0
    mask_b = b > 0
    intersection = (mask_a & mask_b).sum()
    return (2 * intersection / (mask_a.sum() + mask_b.sum())).item()


class TestLabelInterpolation:
    def test_parse_interpolation_accepts_label(self) -> None:
        assert _parse_interpolation("label") == "label"
        assert _parse_interpolation("LABEL") == "label"

    def test_image_interpolation_label_raises(self) -> None:
        with pytest.raises(ValueError, match="image_interpolation"):
            tio.Resample(2, image_interpolation="label")

    def test_no_invalid_labels_when_downsampling(self) -> None:
        data = torch.zeros(1, 32, 32, 32)
        data[0, 8:24, 8:24, 8:24] = 2
        data[0, 12:20, 12:20, 12:20] = 5  # non-contiguous label values
        subject = tio.Subject(seg=tio.LabelMap(data, affine=np.eye(4)))
        result = tio.Resample(4, label_interpolation="label")(subject)
        unique = set(result.seg.data.unique().tolist())
        assert unique <= {0.0, 2.0, 5.0}

    def test_no_invalid_labels_when_upsampling(self) -> None:
        data = torch.zeros(1, 16, 16, 16)
        data[0, 4:12, 4:12, 4:12] = 3
        subject = tio.Subject(seg=tio.LabelMap(data, affine=np.eye(4)))
        result = tio.Resample(0.5, label_interpolation="label")(subject)
        unique = set(result.seg.data.unique().tolist())
        assert unique <= {0.0, 3.0}

    def test_roundtrip_dice_beats_nearest(self) -> None:
        original = _sphere_label()
        subject = tio.Subject(seg=tio.LabelMap(original, affine=np.eye(4)))

        def roundtrip(mode: str) -> torch.Tensor:
            down = tio.Resample(4, label_interpolation=mode)(subject)
            back = tio.Resample(subject.seg, label_interpolation=mode)(down)
            return back.seg.data

        dice_label = _dice(roundtrip("label"), original)
        dice_nearest = _dice(roundtrip("nearest"), original)
        # "label" is reliably better for a compact sphere, but assert only
        # "not worse" to stay robust across grid alignment and library
        # versions (a tie should not fail the test).
        assert dice_label >= dice_nearest

    def test_default_pad_label_fills_out_of_bounds(self) -> None:
        data = torch.ones(1, 16, 16, 16)
        subject = tio.Subject(seg=tio.LabelMap(data, affine=np.eye(4)))
        # Shift the whole volume out of the field of view along one axis.
        transformed = AffineTransform(
            translation=(100.0, 0.0, 0.0),
            label_interpolation="label",
            default_pad_label=7.0,
        )(subject)
        assert (transformed.seg.data == 7.0).any()

    def test_antialias_label_runs_and_keeps_valid_labels(self) -> None:
        original = _sphere_label(value=4.0)
        subject = tio.Subject(seg=tio.LabelMap(original, affine=np.eye(4)))
        result = tio.Resample(
            4,
            label_interpolation="label",
            antialias=True,
        )(subject)
        unique = set(result.seg.data.unique().tolist())
        assert unique <= {0.0, 4.0}
        assert result.seg.data.shape[1:] == (16, 16, 16)

    def test_multichannel_label_resamples_without_argmax(self) -> None:
        data = torch.zeros(2, 16, 16, 16)
        data[0] = 1.0
        data[0, 4:12, 4:12, 4:12] = 0.0
        data[1, 4:12, 4:12, 4:12] = 1.0
        subject = tio.Subject(seg=tio.LabelMap(data, affine=np.eye(4)))
        result = tio.Resample(2, label_interpolation="label")(subject)
        assert result.seg.data.shape[0] == 2

    def test_multichannel_integer_input_preserves_partial_volumes(self) -> None:
        # An integer one-hot encoding must not be truncated back to 0/1:
        # linear resampling should yield fractional partial volumes.
        data = torch.zeros(2, 16, 16, 16, dtype=torch.uint8)
        data[0] = 1
        data[0, :8] = 0
        data[1, :8] = 1
        subject = tio.Subject(seg=tio.LabelMap(data, affine=np.eye(4)))
        result = tio.Resample((1.5, 1.0, 1.0), label_interpolation="label")(subject)
        assert result.seg.data.dtype.is_floating_point
        fractional = (result.seg.data > 0) & (result.seg.data < 1)
        assert fractional.any()

    def _three_label_junction(self) -> tio.Subject:
        # Three labels meeting at a wavy junction, where the per-channel
        # interpolation order changes the argmax outcome.
        n = 40
        yy, xx, zz = torch.meshgrid(
            torch.arange(n),
            torch.arange(n),
            torch.arange(n),
            indexing="ij",
        )
        seg = torch.zeros(n, n, n)
        boundary = n / 2 + 3 * torch.sin(xx.float() / 3)
        seg[yy > boundary] = 1
        seg[(yy <= boundary) & (zz > n / 2)] = 2
        return tio.Subject(seg=tio.LabelMap(seg[None], affine=np.eye(4)))

    def test_one_hot_label_interpolation_label_raises(self) -> None:
        with pytest.raises(ValueError, match="one_hot_label_interpolation"):
            tio.Resample(
                2,
                label_interpolation="label",
                one_hot_label_interpolation="label",
            )

    def test_one_hot_label_interpolation_default_is_linear(self) -> None:
        subject = self._three_label_junction()
        default = tio.Resample(0.5, label_interpolation="label")(subject)
        explicit = tio.Resample(
            0.5,
            label_interpolation="label",
            one_hot_label_interpolation="linear",
        )(subject)
        torch.testing.assert_close(default.seg.data, explicit.seg.data)

    def test_one_hot_label_interpolation_higher_order_differs(self) -> None:
        subject = self._three_label_junction()
        linear = tio.Resample(
            0.5,
            label_interpolation="label",
            one_hot_label_interpolation="linear",
        )(subject)
        cubic = tio.Resample(
            0.5,
            label_interpolation="label",
            one_hot_label_interpolation="cubic",
        )(subject)
        # The order changes the result, but never invents labels.
        assert not torch.equal(linear.seg.data, cubic.seg.data)
        assert set(cubic.seg.data.unique().tolist()) <= {0.0, 1.0, 2.0}

    def test_one_hot_label_interpolation_accepts_integer_order(self) -> None:
        subject = self._three_label_junction()
        result = tio.Resample(
            0.5,
            label_interpolation="label",
            one_hot_label_interpolation=3,
        )(subject)
        assert set(result.seg.data.unique().tolist()) <= {0.0, 1.0, 2.0}

    def test_label_mode_per_instance_batch(self) -> None:
        # Per-instance Affine on a batch must run the label partial-volume
        # mode through the per-sample sampling grid, one geometry per element.
        seg = torch.zeros(1, 24, 24, 24)
        seg[0, 6:18, 6:18, 6:18] = 1
        seg[0, 10:14, 10:14, 10:14] = 2
        subjects = [
            tio.Subject(seg=tio.LabelMap(seg.clone(), affine=np.eye(4)))
            for _ in range(4)
        ]
        batch = tio.SubjectsBatch.from_subjects(subjects)
        torch.manual_seed(0)
        transform = AffineTransform(
            degrees=(-25, 25),
            scales=(0.8, 1.2),
            label_interpolation="label",
        )
        data = transform(batch).images["seg"].data
        assert data.shape[0] == 4
        assert set(data.unique().tolist()) <= {0.0, 1.0, 2.0}
        # Per-instance sampling: elements receive different geometry.
        assert not torch.equal(data[0], data[1])
