"""Tests for BoundingBoxes."""

from __future__ import annotations

import copy

import numpy as np
import numpy.typing as npt
import pytest
import torch
from nibabel.affines import apply_affine

from torchio.data.bboxes import BoundingBoxes
from torchio.data.bboxes import BoundingBoxFormat
from torchio.data.bboxes import Representation


class TestBoundingBoxFormat:
    """Test the format class."""

    def test_ijk_corners(self):
        fmt = BoundingBoxFormat("IJK", Representation.CORNERS)
        assert fmt.axes == "IJK"
        assert fmt.representation == Representation.CORNERS

    def test_ras_center_size(self):
        fmt = BoundingBoxFormat("RAS", Representation.CENTER_SIZE)
        assert fmt.axes == "RAS"
        assert fmt.representation == Representation.CENTER_SIZE

    def test_string_representation(self):
        fmt = BoundingBoxFormat("IJK", "corners")
        assert fmt.representation == Representation.CORNERS

    def test_string_center_size(self):
        fmt = BoundingBoxFormat("IJK", "center_size")
        assert fmt.representation == Representation.CENTER_SIZE

    def test_invalid_axes_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            BoundingBoxFormat("XYZ", Representation.CORNERS)

    def test_equality(self):
        a = BoundingBoxFormat("IJK", Representation.CORNERS)
        b = BoundingBoxFormat("IJK", Representation.CORNERS)
        assert a == b

    def test_inequality_axes(self):
        a = BoundingBoxFormat("IJK", Representation.CORNERS)
        b = BoundingBoxFormat("KJI", Representation.CORNERS)
        assert a != b

    def test_inequality_representation(self):
        a = BoundingBoxFormat("IJK", Representation.CORNERS)
        b = BoundingBoxFormat("IJK", Representation.CENTER_SIZE)
        assert a != b

    def test_hashable(self):
        a = BoundingBoxFormat("IJK", Representation.CORNERS)
        b = BoundingBoxFormat("IJK", Representation.CORNERS)
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_repr(self):
        fmt = BoundingBoxFormat("RAS", Representation.CORNERS)
        r = repr(fmt)
        assert "RAS" in r
        assert "corners" in r

    def test_predefined_ijkijk(self):
        assert (
            BoundingBoxFormat(
                "IJK",
                Representation.CORNERS,
            )
            == BoundingBoxFormat.IJKIJK
        )

    def test_predefined_ijkwhd(self):
        assert (
            BoundingBoxFormat(
                "IJK",
                Representation.CENTER_SIZE,
            )
            == BoundingBoxFormat.IJKWHD
        )


class TestBoundingBoxesCreation:
    def test_from_tensor(self):
        data = torch.tensor([[10, 20, 30, 50, 60, 70]], dtype=torch.float32)
        bboxes = BoundingBoxes(data, format=BoundingBoxFormat.IJKIJK)
        assert bboxes.data.shape == (1, 6)

    def test_from_numpy(self):
        data = np.array([[10, 20, 30, 50, 60, 70]], dtype=np.float32)
        bboxes = BoundingBoxes(data, format=BoundingBoxFormat.IJKIJK)
        assert isinstance(bboxes.data, torch.Tensor)

    def test_multiple_boxes(self):
        data = torch.tensor(
            [
                [10, 20, 30, 50, 60, 70],
                [0, 0, 0, 10, 10, 10],
            ],
            dtype=torch.float32,
        )
        bboxes = BoundingBoxes(data, format=BoundingBoxFormat.IJKIJK)
        assert len(bboxes) == 2

    def test_empty_boxes(self):
        bboxes = BoundingBoxes(
            torch.zeros(0, 6),
            format=BoundingBoxFormat.IJKIJK,
        )
        assert len(bboxes) == 0

    def test_with_labels(self):
        data = torch.tensor([[0, 0, 0, 10, 10, 10]], dtype=torch.float32)
        labels = torch.tensor([3])
        bboxes = BoundingBoxes(
            data,
            format=BoundingBoxFormat.IJKIJK,
            labels=labels,
        )
        assert bboxes.labels is not None
        assert bboxes.labels[0] == 3

    def test_with_affine(self):
        data = torch.tensor([[0, 0, 0, 10, 10, 10]], dtype=torch.float32)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        bboxes = BoundingBoxes(
            data,
            format=BoundingBoxFormat.IJKIJK,
            affine=affine,
        )
        np.testing.assert_array_equal(bboxes.affine.numpy(), affine)

    def test_with_metadata(self):
        data = torch.tensor([[0, 0, 0, 10, 10, 10]], dtype=torch.float32)
        bboxes = BoundingBoxes(
            data,
            format=BoundingBoxFormat.IJKIJK,
            metadata={"source": "manual"},
        )
        assert bboxes.metadata["source"] == "manual"

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="N, 6"):
            BoundingBoxes(
                torch.tensor([[1, 2, 3]]),
                format=BoundingBoxFormat.IJKIJK,
            )

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="N, 6"):
            BoundingBoxes(
                torch.tensor([1, 2, 3, 4, 5, 6]),
                format=BoundingBoxFormat.IJKIJK,
            )

    def test_labels_length_mismatch_raises(self):
        data = torch.tensor([[0, 0, 0, 10, 10, 10]], dtype=torch.float32)
        labels = torch.tensor([1, 2])  # 2 labels for 1 box
        with pytest.raises(ValueError, match="labels"):
            BoundingBoxes(
                data,
                format=BoundingBoxFormat.IJKIJK,
                labels=labels,
            )


class TestBoundingBoxesProperties:
    def test_len(self):
        data = torch.randn(5, 6)
        bboxes = BoundingBoxes(data, format=BoundingBoxFormat.IJKIJK)
        assert len(bboxes) == 5

    def test_num_boxes(self):
        data = torch.randn(3, 6)
        bboxes = BoundingBoxes(data, format=BoundingBoxFormat.IJKIJK)
        assert bboxes.num_boxes == 3


class TestRepresentationConversion:
    """Test corners ↔ center_size conversion (same axes)."""

    def test_corners_to_center_size(self):
        # Box from (10, 20, 30) to (50, 60, 70)
        # Center: (30, 40, 50), Size: (40, 40, 40)
        data = torch.tensor([[10, 20, 30, 50, 60, 70]], dtype=torch.float32)
        bboxes = BoundingBoxes(data, format=BoundingBoxFormat.IJKIJK)
        converted = bboxes.to_format(BoundingBoxFormat.IJKWHD)
        expected = torch.tensor([[30, 40, 50, 40, 40, 40]], dtype=torch.float32)
        torch.testing.assert_close(converted.data, expected)
        assert converted.format == BoundingBoxFormat.IJKWHD

    def test_center_size_to_corners(self):
        data = torch.tensor([[30, 40, 50, 40, 40, 40]], dtype=torch.float32)
        bboxes = BoundingBoxes(data, format=BoundingBoxFormat.IJKWHD)
        converted = bboxes.to_format(BoundingBoxFormat.IJKIJK)
        expected = torch.tensor([[10, 20, 30, 50, 60, 70]], dtype=torch.float32)
        torch.testing.assert_close(converted.data, expected)

    def test_same_format_noop(self):
        data = torch.tensor([[10, 20, 30, 50, 60, 70]], dtype=torch.float32)
        bboxes = BoundingBoxes(data, format=BoundingBoxFormat.IJKIJK)
        converted = bboxes.to_format(BoundingBoxFormat.IJKIJK)
        torch.testing.assert_close(converted.data, data)

    def test_roundtrip(self):
        data = torch.tensor(
            [
                [10, 20, 30, 50, 60, 70],
                [0, 0, 0, 100, 100, 100],
            ],
            dtype=torch.float32,
        )
        bboxes = BoundingBoxes(data, format=BoundingBoxFormat.IJKIJK)
        roundtrip = bboxes.to_format(BoundingBoxFormat.IJKWHD).to_format(
            BoundingBoxFormat.IJKIJK,
        )
        torch.testing.assert_close(roundtrip.data, data)

    def test_preserves_labels(self):
        data = torch.tensor([[10, 20, 30, 50, 60, 70]], dtype=torch.float32)
        labels = torch.tensor([5])
        bboxes = BoundingBoxes(
            data,
            format=BoundingBoxFormat.IJKIJK,
            labels=labels,
        )
        converted = bboxes.to_format(BoundingBoxFormat.IJKWHD)
        assert converted.labels is not None
        assert converted.labels[0] == 5


class TestVoxelAxisPermutation:
    """Test conversion between voxel axis orderings."""

    def test_ijk_to_kji_corners(self):
        # Box: i=[10,50], j=[20,60], k=[30,70]
        data = torch.tensor([[10, 20, 30, 50, 60, 70]], dtype=torch.float32)
        bboxes = BoundingBoxes(data, format=BoundingBoxFormat.IJKIJK)
        target = BoundingBoxFormat("KJI", Representation.CORNERS)
        converted = bboxes.to_format(target)
        # KJI corners: k1, j1, i1, k2, j2, i2
        expected = torch.tensor([[30, 20, 10, 70, 60, 50]], dtype=torch.float32)
        torch.testing.assert_close(converted.data, expected)

    def test_ijk_to_jki_corners(self):
        data = torch.tensor([[10, 20, 30, 50, 60, 70]], dtype=torch.float32)
        bboxes = BoundingBoxes(data, format=BoundingBoxFormat.IJKIJK)
        target = BoundingBoxFormat("JKI", Representation.CORNERS)
        converted = bboxes.to_format(target)
        # JKI corners: j1, k1, i1, j2, k2, i2
        expected = torch.tensor([[20, 30, 10, 60, 70, 50]], dtype=torch.float32)
        torch.testing.assert_close(converted.data, expected)

    def test_ijk_to_kji_center_size(self):
        # center=(30,40,50), size_i=40, size_j=40, size_k=40
        data = torch.tensor([[30, 40, 50, 40, 40, 40]], dtype=torch.float32)
        bboxes = BoundingBoxes(data, format=BoundingBoxFormat.IJKWHD)
        target = BoundingBoxFormat("KJI", Representation.CENTER_SIZE)
        converted = bboxes.to_format(target)
        # KJI center_size: kc, jc, ic, size_k, size_j, size_i
        expected = torch.tensor([[50, 40, 30, 40, 40, 40]], dtype=torch.float32)
        torch.testing.assert_close(converted.data, expected)

    def test_roundtrip_ijk_kji(self):
        data = torch.tensor(
            [[10, 20, 30, 50, 60, 70]],
            dtype=torch.float32,
        )
        bboxes = BoundingBoxes(data, format=BoundingBoxFormat.IJKIJK)
        kji = BoundingBoxFormat("KJI", Representation.CORNERS)
        roundtrip = bboxes.to_format(kji).to_format(BoundingBoxFormat.IJKIJK)
        torch.testing.assert_close(roundtrip.data, data)


class TestAnatomicalAxisConversion:
    """Test conversion between anatomical axis systems."""

    def test_ras_to_lpi_corners(self):
        # Box in RAS: r=[10,50], a=[20,60], s=[30,70]
        data = torch.tensor([[10, 20, 30, 50, 60, 70]], dtype=torch.float32)
        src = BoundingBoxFormat("RAS", Representation.CORNERS)
        tgt = BoundingBoxFormat("LPI", Representation.CORNERS)
        bboxes = BoundingBoxes(data, format=src)
        converted = bboxes.to_format(tgt)
        # L=-R, P=-A, I=-S. Negation swaps min/max for corners.
        # L: [-50, -10], P: [-60, -20], I: [-70, -30]
        expected = torch.tensor(
            [[-50, -60, -70, -10, -20, -30]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(converted.data, expected)

    def test_ras_to_asr_corners(self):
        """Pure reorder, no flips."""
        data = torch.tensor([[10, 20, 30, 50, 60, 70]], dtype=torch.float32)
        src = BoundingBoxFormat("RAS", Representation.CORNERS)
        tgt = BoundingBoxFormat("ASR", Representation.CORNERS)
        bboxes = BoundingBoxes(data, format=src)
        converted = bboxes.to_format(tgt)
        # ASR: a1, s1, r1, a2, s2, r2
        expected = torch.tensor([[20, 30, 10, 60, 70, 50]], dtype=torch.float32)
        torch.testing.assert_close(converted.data, expected)

    def test_ras_to_lpi_center_size(self):
        # Center in RAS: (30, 40, 50), sizes: (40, 40, 40)
        data = torch.tensor([[30, 40, 50, 40, 40, 40]], dtype=torch.float32)
        src = BoundingBoxFormat("RAS", Representation.CENTER_SIZE)
        tgt = BoundingBoxFormat("LPI", Representation.CENTER_SIZE)
        bboxes = BoundingBoxes(data, format=src)
        converted = bboxes.to_format(tgt)
        # Center negated: (-30, -40, -50). Sizes unchanged: (40, 40, 40).
        expected = torch.tensor(
            [[-30, -40, -50, 40, 40, 40]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(converted.data, expected)

    def test_roundtrip_ras_lpi(self):
        data = torch.tensor(
            [[10, 20, 30, 50, 60, 70]],
            dtype=torch.float32,
        )
        src = BoundingBoxFormat("RAS", Representation.CORNERS)
        tgt = BoundingBoxFormat("LPI", Representation.CORNERS)
        bboxes = BoundingBoxes(data, format=src)
        roundtrip = bboxes.to_format(tgt).to_format(src)
        torch.testing.assert_close(roundtrip.data, data)


class TestVoxelAnatomicalConversion:
    """Test conversion between voxel and anatomical using the affine."""

    def test_ijk_to_ras_identity_affine(self):
        """With identity affine, IJK == RAS numerically."""
        data = torch.tensor([[10, 20, 30, 50, 60, 70]], dtype=torch.float32)
        bboxes = BoundingBoxes(
            data,
            format=BoundingBoxFormat.IJKIJK,
        )  # identity affine
        tgt = BoundingBoxFormat("RAS", Representation.CORNERS)
        converted = bboxes.to_format(tgt)
        torch.testing.assert_close(converted.data, data)

    def test_ijk_to_ras_with_spacing(self):
        """Spacing scales coordinates."""
        data = torch.tensor([[0, 0, 0, 10, 10, 10]], dtype=torch.float32)
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        bboxes = BoundingBoxes(
            data,
            format=BoundingBoxFormat.IJKIJK,
            affine=affine,
        )
        tgt = BoundingBoxFormat("RAS", Representation.CORNERS)
        converted = bboxes.to_format(tgt)
        expected = torch.tensor([[0, 0, 0, 20, 30, 40]], dtype=torch.float32)
        torch.testing.assert_close(converted.data, expected)

    def test_ijk_to_ras_with_origin(self):
        """Origin shifts coordinates."""
        data = torch.tensor([[0, 0, 0, 10, 10, 10]], dtype=torch.float32)
        affine = np.eye(4)
        affine[:3, 3] = [100, 200, 300]
        bboxes = BoundingBoxes(
            data,
            format=BoundingBoxFormat.IJKIJK,
            affine=affine,
        )
        tgt = BoundingBoxFormat("RAS", Representation.CORNERS)
        converted = bboxes.to_format(tgt)
        expected = torch.tensor(
            [[100, 200, 300, 110, 210, 310]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(converted.data, expected)

    def test_ras_to_ijk_roundtrip(self):
        data = torch.tensor([[5, 10, 15, 25, 30, 35]], dtype=torch.float32)
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        affine[:3, 3] = [10, 20, 30]
        bboxes = BoundingBoxes(
            data,
            format=BoundingBoxFormat.IJKIJK,
            affine=affine,
        )
        ras_fmt = BoundingBoxFormat("RAS", Representation.CORNERS)
        roundtrip = bboxes.to_format(ras_fmt).to_format(BoundingBoxFormat.IJKIJK)
        torch.testing.assert_close(roundtrip.data, data, atol=1e-5, rtol=1e-5)

    def test_ijk_to_lpi(self):
        """Voxel to non-RAS anatomical (combines affine + flip)."""
        data = torch.tensor([[0, 0, 0, 10, 10, 10]], dtype=torch.float32)
        # Identity affine → world is RAS
        bboxes = BoundingBoxes(
            data,
            format=BoundingBoxFormat.IJKIJK,
        )
        tgt = BoundingBoxFormat("LPI", Representation.CORNERS)
        converted = bboxes.to_format(tgt)
        # In RAS: [0,0,0,10,10,10]. LPI = negate all, swap corners.
        expected = torch.tensor(
            [[-10, -10, -10, 0, 0, 0]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(converted.data, expected)

    def test_no_affine_cross_type_raises(self):
        """Voxel↔anatomical conversion requires an affine, but identity
        is the default so it should work. This test documents that the
        affine is used implicitly."""
        data = torch.tensor([[0, 0, 0, 10, 10, 10]], dtype=torch.float32)
        bboxes = BoundingBoxes(data, format=BoundingBoxFormat.IJKIJK)
        tgt = BoundingBoxFormat("RAS", Representation.CORNERS)
        # Should work (uses default identity affine)
        converted = bboxes.to_format(tgt)
        assert converted.format == tgt

    @pytest.mark.parametrize("voxel_axes", ["IJK", "KJI"])
    @pytest.mark.parametrize(
        ("anatomical_axes", "order", "signs"),
        [
            ("RAS", (0, 1, 2), (1, 1, 1)),
            ("LPI", (0, 1, 2), (-1, -1, -1)),
            ("ASR", (1, 2, 0), (1, 1, 1)),
        ],
    )
    @pytest.mark.parametrize("representation", list(Representation))
    @pytest.mark.parametrize("inverse", [False, True])
    def test_cross_type_uses_ras_world(
        self,
        annotation_affine: npt.NDArray[np.float64],
        voxel_axes: str,
        anatomical_axes: str,
        order: tuple[int, int, int],
        signs: tuple[int, int, int],
        representation: Representation,
        inverse: bool,
    ) -> None:
        ijk = np.array([[1, 2, 3, 4, 6, 8], [-3, -2, -1, 0, 1, 2]], dtype=float)
        world = apply_affine(annotation_affine, ijk.reshape(-1, 2, 3))
        anatomical_corners = world[:, :, order] * signs
        anatomical = np.concatenate(
            [anatomical_corners.min(axis=1), anatomical_corners.max(axis=1)], axis=1
        )
        indices = ["IJK".index(axis) for axis in voxel_axes]
        voxel = ijk[:, indices + [index + 3 for index in indices]]
        if representation == Representation.CENTER_SIZE:
            voxel = np.concatenate(
                [(voxel[:, :3] + voxel[:, 3:]) / 2, voxel[:, 3:] - voxel[:, :3]], axis=1
            )
            anatomical = np.concatenate(
                [
                    (anatomical[:, :3] + anatomical[:, 3:]) / 2,
                    anatomical[:, 3:] - anatomical[:, :3],
                ],
                axis=1,
            )
        if inverse:
            data, expected = anatomical, voxel
            source, target = anatomical_axes, voxel_axes
        else:
            data, expected = voxel, anatomical
            source, target = voxel_axes, anatomical_axes
        labels = torch.tensor([1, 2])
        boxes = BoundingBoxes(
            data,
            format=BoundingBoxFormat(source, representation),
            affine=annotation_affine,
            labels=labels,
            metadata={"name": "regions"},
        )
        target_format = BoundingBoxFormat(target, representation)

        converted = boxes.to_format(target_format)

        torch.testing.assert_close(converted.data, torch.from_numpy(expected).float())
        assert converted.format == target_format
        np.testing.assert_array_equal(converted.affine.numpy(), annotation_affine)
        torch.testing.assert_close(converted.labels, labels)
        assert converted.metadata == boxes.metadata
        torch.testing.assert_close(boxes.data, torch.from_numpy(data).float())


class TestNewLike:
    def test_new_like_preserves_format(self):
        bboxes = BoundingBoxes(
            torch.randn(3, 6),
            format=BoundingBoxFormat.IJKWHD,
        )
        new = bboxes.new_like(data=torch.randn(2, 6))
        assert new.format == BoundingBoxFormat.IJKWHD

    def test_new_like_preserves_affine(self):
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        bboxes = BoundingBoxes(
            torch.randn(3, 6),
            format=BoundingBoxFormat.IJKIJK,
            affine=affine,
        )
        new = bboxes.new_like(data=torch.randn(1, 6))
        np.testing.assert_array_equal(new.affine.numpy(), affine)

    def test_new_like_preserves_metadata(self):
        bboxes = BoundingBoxes(
            torch.randn(3, 6),
            format=BoundingBoxFormat.IJKIJK,
            metadata={"organ": "liver"},
        )
        new = bboxes.new_like(data=torch.randn(1, 6))
        assert new.metadata["organ"] == "liver"


class TestRepr:
    def test_repr_contains_info(self):
        data = torch.randn(3, 6)
        bboxes = BoundingBoxes(data, format=BoundingBoxFormat.IJKIJK)
        r = repr(bboxes)
        assert "BoundingBoxes" in r
        assert "3" in r
        assert "IJK" in r


class TestCopy:
    def test_deepcopy(self):
        bboxes = BoundingBoxes(
            torch.randn(3, 6),
            format=BoundingBoxFormat.IJKIJK,
            labels=torch.tensor([1, 2, 3]),
            metadata={"a": 1},
        )
        copied = copy.deepcopy(bboxes)
        assert torch.equal(copied.data, bboxes.data)
        assert torch.equal(copied.labels, bboxes.labels)
        copied.metadata["a"] = 2
        assert bboxes.metadata["a"] == 1
