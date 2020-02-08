from typing import Callable, Sequence, Optional
import torch
from ..torchio import DATA, TypeCallable
from ..utils import is_image_dict
from .transform import Transform


class Lambda(Transform):
    def __init__(
            self,
            function: TypeCallable,
            types_to_apply: Optional[Sequence[str]] = None,
            verbose: bool = False,
            ):
        super().__init__(verbose=verbose)
        self.function = function
        self.types_to_apply = types_to_apply

    def apply_transform(self, sample: dict) -> dict:
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue

            image_type = image_dict['type']
            if self.types_to_apply is not None:
                if image_type not in self.types_to_apply:
                    continue

            function_arg = image_dict[DATA][0]
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
            image_dict[DATA][0] = result
        return sample
