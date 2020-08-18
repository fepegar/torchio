from typing import Dict, Optional

import torch

from ...data.subject import Subject
from ...torchio import TypePatchSize, DATA, TYPE, LABEL
from .weighted import WeightedSampler


class LabelSampler(WeightedSampler):
    r"""Extract random patches with labeled voxels at their center.

    This sampler yields patches whose center value is greater than 0
    in the :py:attr:`label_name`.

    Args:
        patch_size: See :py:class:`~torchio.data.PatchSampler`.
        label_name: Name of the label image in the sample that will be used to
            generate the sampling probability map. If ``None``, the first image
            of type :py:attr:`torchio.LABEL` found in the subject sample will be
            used.
        label_probabilities: Dictionary containing the probability that each
            class will be sampled. Probabilities do not need to be normalized.
            For example, a value of ``{0: 0, 1: 2, 2: 1, 3: 1}`` will create a
            sampler whose patches centers will have 50% probability of being
            labeled as ``1``, 25% of being ``2`` and 25% of being ``3``.
            If ``None``, the label map is binarized and the value is set to
            ``{0: 0, 1: 1}``.

    Example:
        >>> import torchio
        >>> subject = torchio.datasets.Colin27()
        >>> subject
        Colin27(Keys: ('t1', 'head', 'brain'); images: 3)
        >>> sample = torchio.SubjectsDataset([subject])[0]
        >>> sampler = torchio.data.LabelSampler(64, 'brain')
        >>> generator = sampler(sample)
        >>> for patch in generator:
        ...     print(patch.shape)

    If you want a specific number of patches from a volume, e.g. 10:

        >>> generator = sampler(sample, num_patches=10)
        >>> for patch in iterator:
        ...     print(patch.shape)

    """
    def __init__(
            self,
            patch_size: TypePatchSize,
            label_name: Optional[str] = None,
            label_probabilities: Optional[Dict[int, float]] = None,
        ):
        super().__init__(patch_size, probability_map=label_name)
        self.label_probabilities_dict = label_probabilities

    def get_probability_map(self, sample: Subject) -> torch.Tensor:
        if self.probability_map_name is None:
            for image in sample.get_images(intensity_only=False):
                if image[TYPE] == LABEL:
                    label_map_tensor = image[DATA]
                    break
        elif self.probability_map_name in sample:
            label_map_tensor = sample[self.probability_map_name][DATA]
        else:
            message = (
                f'Image "{self.probability_map_name}"'
                f' not found in subject sample: {sample}'
            )
            raise KeyError(message)
        if self.label_probabilities_dict is None:
            return label_map_tensor > 0
        probability_map = self.get_probabilities_from_label_map(
            label_map_tensor,
            self.label_probabilities_dict,
        )
        return probability_map

    @staticmethod
    def get_probabilities_from_label_map(
            label_map: torch.Tensor,
            label_probabilities_dict: Dict[int, float],
            ) -> torch.Tensor:
        """Create probability map according to label map probabilities."""
        probability_map = torch.zeros_like(label_map)
        label_probs = torch.Tensor(list(label_probabilities_dict.values()))
        normalized_probs = label_probs / label_probs.sum()
        iterable = zip(label_probabilities_dict, normalized_probs)
        for label, label_probability in iterable:
            mask = label_map == label
            label_size = mask.sum()
            if not label_size:
                continue
            prob_voxels = label_probability / label_size
            probability_map[mask] = prob_voxels
        return probability_map
