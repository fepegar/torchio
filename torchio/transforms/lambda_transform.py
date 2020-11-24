from typing import Sequence, Optional
import torch
from ..data.subject import Subject
from ..torchio import DATA, TYPE, TypeCallable
from .transform import Transform


class Lambda(Transform):
    """Applies a user-defined function as transform.

    Args:
        function: Callable that receives and returns a 4D
            :class:`torch.Tensor`.
        types_to_apply: List of strings corresponding to the image types to
            which this transform should be applied. If ``None``, the transform
            will be applied to all images in the subject.
        p: Probability that this transform will be applied.
        keys: See :class:`~torchio.transforms.Transform`.

    Example:
        >>> import torchio as tio
        >>> invert_intensity = tio.Lambda(lambda x: -x, types_to_apply=[tio.INTENSITY])
        >>> invert_mask = tio.Lambda(lambda x: 1 - x, types_to_apply=[tio.LABEL])
        >>> def double(x):
        ...     return 2 * x
        >>> double_transform = tio.Lambda(double)
    """
    def __init__(
            self,
            function: TypeCallable,
            types_to_apply: Optional[Sequence[str]] = None,
            p: float = 1,
            keys: Optional[Sequence[str]] = None,
            ):
        super().__init__(p=p, keys=keys)
        self.function = function
        self.types_to_apply = types_to_apply
        self.args_names = 'function', 'types_to_apply'

    def apply_transform(self, subject: Subject) -> Subject:
        for image in subject.get_images(intensity_only=False):

            image_type = image[TYPE]
            if self.types_to_apply is not None:
                if image_type not in self.types_to_apply:
                    continue

            function_arg = image[DATA]
            result = self.function(function_arg)
            if not isinstance(result, torch.Tensor):
                message = (
                    'The returned value from the callable argument must be'
                    f' of type {torch.Tensor}, not {type(result)}'
                )
                raise ValueError(message)
            if result.dtype != torch.float32:
                message = (
                    'The data type of the returned value must be'
                    f' of type {torch.float32}, not {result.dtype}'
                )
                raise ValueError(message)
            if result.ndim != function_arg.ndim:
                message = (
                    'The number of dimensions of the returned value must'
                    f' be {function_arg.ndim}, not {result.ndim}'
                )
                raise ValueError(message)
            image[DATA] = result
        return subject
