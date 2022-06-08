import torch
from ....data.subject import Subject
from .normalization_transform import NormalizationTransform, TypeMaskingMethod


class ZNormalization(NormalizationTransform):
    """Subtract mean and divide by standard deviation.

    Args:
        masking_method: See
            :class:`~torchio.transforms.preprocessing.intensity.NormalizationTransform`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """
    def __init__(
            self,
            masking_method: TypeMaskingMethod = None,
            **kwargs
            ):
        super().__init__(masking_method=masking_method, **kwargs)
        self.args_names = ('masking_method',)

    def apply_normalization(
            self,
            subject: Subject,
            image_name: str,
            mask: torch.Tensor,
            ) -> None:
        image = subject[image_name]
        standardized = self.znorm(
            image.data,
            mask,
        )
        if standardized is None:
            message = (
                'Standard deviation is 0 for masked values'
                f' in image "{image_name}" ({image.path})'
            )
            raise RuntimeError(message)
        image.set_data(standardized)

    @staticmethod
    def znorm(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tensor = tensor.clone().float()
        values = tensor.masked_select(mask)
        mean, std = values.mean(), values.std()
        if std == 0:
            return None
        tensor -= mean
        tensor /= std
        return tensor
