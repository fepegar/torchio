import torch
from ...data.subject import Subject
from ...torchio import TypePatchSize
from .weighted import WeightedSampler


class LabelSampler(WeightedSampler):
    r"""Extract random patches with labeled voxels at their center.

    This sampler yields patches whose center value is greater than 0
    in the :py:attr:`label_name`.

    Args:
        patch_size: See :py:class:`~torchio.data.PatchSampler`.
        label_name: Name of the label image in the sample that will be used to
            generate the sampling probability map.

    Example:
        >>> import torchio
        >>> subject = torchio.datasets.Colin27()
        >>> subject
        Colin27(Keys: ('t1', 'head', 'brain'); images: 3)
        >>> sample = torchio.ImagesDataset([subject])[0]
        >>> sampler = torchio.data.LabelSampler(64, 'brain')
        >>> generator = sampler(sample)
        >>> for patch in generator:
        ...     print(patch.shape)

    If you want a specific number of patches from a volume, e.g. 10:

        >>> generator = sampler(sample, num_patches=10)
        >>> for patch in iterator:
        ...     print(patch.shape)

    """
    def __init__(self, patch_size: TypePatchSize, label_name: str):
        super().__init__(patch_size, probability_map=label_name)

    def get_probability_map(self, sample: Subject) -> torch.Tensor:
        """Return binarized image for sampling."""
        if self.probability_map_name in sample:
            data = sample[self.probability_map_name].data > 0.5
        else:
            message = (
                f'Image "{self.probability_map_name}"'
                f' not found in subject sample: {sample}'
            )
            raise KeyError(message)
        return data
