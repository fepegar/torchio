"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch

from torchio.data.batch import SubjectsBatch
from torchio.data.batch import _slice_params


@pytest.fixture(params=["RAS", "LAS", "ASL"])
def annotation_affine(request: pytest.FixtureRequest) -> npt.NDArray[np.float64]:
    linear = {
        "RAS": [[2, 0, 0], [0, 3, 0], [0, 0, 4]],
        "LAS": [[-2, 0, 0], [0, 3, 0], [0, 0, 4]],
        "ASL": [[0, 0, -4], [2, 0, 0], [0, 3, 0]],
    }
    affine = np.eye(4)
    affine[:3, :3] = linear[request.param]
    affine[:3, 3] = [10, 20, 30]
    return affine


@pytest.fixture
def assert_vectorized() -> Callable[..., None]:
    """Return an assertion that a per-instance transform is vectorized faithfully.

    The returned helper applies *transform* to a batch (per-instance) and then,
    for every element, re-applies the same per-element parameters to that
    element alone. The two must match, which proves the vectorized whole-batch
    computation is equivalent to processing each element independently (no
    cross-element contamination or broadcasting mistakes).

    This is only valid for transforms whose `apply_transform` is deterministic
    given the recorded parameters (the randomness lives in `make_params`).
    Transforms that sample inside `apply_transform` (e.g. Noise, LabelsToImage)
    need a different check.
    """

    def _assert(
        transform: Any,
        batch: SubjectsBatch,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-6,
    ) -> None:
        original = copy.deepcopy(batch)
        result = transform(batch)
        params = result.applied_transforms[-1].params
        assert "_batched_keys" in params, "per-instance path was not active"
        batched_keys = params["_batched_keys"]
        keep = params.get("_keep")
        image_names = list(transform._get_images(result).keys())
        result_images = transform._get_images(result)
        original_subjects = original.unbatch()
        for index in range(original.batch_size):
            single = SubjectsBatch.from_subjects([original_subjects[index]])
            single_input = {
                name: image.data.clone()
                for name, image in transform._get_images(single).items()
            }
            element_params = _slice_params(params, index, batched_keys)
            single = transform.apply_transform(single, element_params)
            single_images = transform._get_images(single)
            gated_out = keep is not None and not keep[index]
            for name in image_names:
                result_row = result_images[name].data[index : index + 1]
                torch.testing.assert_close(
                    result_row,
                    single_images[name].data,
                    rtol=rtol,
                    atol=atol,
                )
                if gated_out:
                    # A gated-out element must be a bit-for-bit no-op.
                    torch.testing.assert_close(
                        result_row,
                        single_input[name],
                        rtol=0,
                        atol=0,
                    )

    return _assert
