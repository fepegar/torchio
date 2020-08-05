from typing import Sequence, Optional, List
import torch
from ..data.subject import Subject
from ..torchio import DATA, TYPE, TypeCallable
from .transform import Transform


class Lambda(Transform):
    """Applies a user-defined function as transform.

    Args:
        function: Callable that receives and returns a 4D
            :py:class:`torch.Tensor`.
        types_to_apply: List of strings corresponding to the image types to
            which this transform should be applied. If ``None``, the transform
            will be applied to all images in the sample.
        p: Probability that this transform will be applied.
        keys: See :py:class:`~torchio.transforms.Transform`.

    Example:
        >>> import torchio
        >>> from torchio.transforms import Lambda
        >>> invert_intensity = Lambda(lambda x: -x, types_to_apply=[torchio.INTENSITY])
        >>> invert_mask = Lambda(lambda x: 1 - x, types_to_apply=[torchio.LABEL])
        >>> def double(x):
        ...     return 2 * x
        >>> double_transform = Lambda(double)
    """
    def __init__(
            self,
            function: TypeCallable,
            types_to_apply: Optional[Sequence[str]] = None,
            p: float = 1,
            keys: Optional[List[str]] = None,
            ):
        super().__init__(p=p, keys=keys)
        self.function = function
        self.types_to_apply = types_to_apply

    def apply_transform(self, sample: Subject) -> dict:
        for image in sample.get_images(intensity_only=False):

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
        return sample
