from typing import Sequence, Optional
import torch
from ..typing import TypeCallable
from ..data.subject import Subject
from ..constants import TYPE
from .transform import Transform


class Lambda(Transform):
    """Applies a user-defined function as transform.

    Args:
        function: Callable that receives and returns a 4D
            :class:`torch.Tensor`.
        types_to_apply: List of strings corresponding to the image types to
            which this transform should be applied. If ``None``, the transform
            will be applied to all images in the subject.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torchio as tio
        >>> invert_intensity = tio.Lambda(lambda x: -x, types_to_apply=[tio.INTENSITY])
        >>> invert_mask = tio.Lambda(lambda x: 1 - x, types_to_apply=[tio.LABEL])
        >>> def double(x):
        ...     return 2 * x
        >>> double_transform = tio.Lambda(double)
    """  # noqa: E501

    def __init__(
            self,
            function: TypeCallable,
            types_to_apply: Optional[Sequence[str]] = None,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.function = function
        self.types_to_apply = types_to_apply
        self.args_names = 'function', 'types_to_apply'

    def apply_transform(self, subject: Subject) -> Subject:
        images = subject.get_images(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )
        for image in images:
            image_type = image[TYPE]
            if self.types_to_apply is not None:
                if image_type not in self.types_to_apply:
                    continue

            function_arg = image.data
            result = self.function(function_arg)
            if not isinstance(result, torch.Tensor):
                message = (
                    'The returned value from the callable argument must be'
                    f' of type {torch.Tensor}, not {type(result)}'
                )
                raise ValueError(message)
            if result.ndim != function_arg.ndim:
                message = (
                    'The number of dimensions of the returned value must'
                    f' be {function_arg.ndim}, not {result.ndim}'
                )
                raise ValueError(message)
            image.set_data(result)
        return subject
