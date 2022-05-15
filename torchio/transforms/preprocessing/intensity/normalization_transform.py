import torch
from ....data.subject import Subject
from ....transforms.transform import TypeMaskingMethod
from ... import IntensityTransform


class NormalizationTransform(IntensityTransform):
    """Base class for intensity preprocessing transforms.

    Args:
        masking_method: Defines the mask used to compute the normalization statistics. It can be one of:

            - ``None``: the mask image is all ones, i.e. all values in the image are used.

            - A string: key to a :class:`torchio.LabelMap` in the subject which is used as a mask,
              OR an anatomical label: ``'Left'``, ``'Right'``, ``'Anterior'``, ``'Posterior'``,
              ``'Inferior'``, ``'Superior'`` which specifies a side of the mask volume to be ones.

            - A function: the mask image is computed as a function of the intensity image.
              The function must receive and return a :class:`torch.Tensor`

        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torchio as tio
        >>> subject = tio.datasets.Colin27()
        >>> subject
        Colin27(Keys: ('t1', 'head', 'brain'); images: 3)
        >>> transform = tio.ZNormalization()  # ZNormalization is a subclass of NormalizationTransform
        >>> transformed = transform(subject)  # use all values to compute mean and std
        >>> transform = tio.ZNormalization(masking_method='brain')
        >>> transformed = transform(subject)  # use only values within the brain
        >>> transform = tio.ZNormalization(masking_method=lambda x: x > x.mean())
        >>> transformed = transform(subject)  # use values above the image mean

    """  # noqa: E501
    def __init__(
            self,
            masking_method: TypeMaskingMethod = None,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.masking_method = masking_method

    def apply_transform(self, subject: Subject) -> Subject:
        for image_name, image in self.get_images_dict(subject).items():
            mask = self.get_mask_from_masking_method(
                self.masking_method,
                subject,
                image.data,
            )
            self.apply_normalization(subject, image_name, mask)
        return subject

    def apply_normalization(
            self,
            subject: Subject,
            image_name: str,
            mask: torch.Tensor,
            ) -> None:
        # There must be a nicer way of doing this
        raise NotImplementedError
