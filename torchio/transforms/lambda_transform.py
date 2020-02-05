import torch
from ..torchio import DATA
from ..utils import is_image_dict
from .transform import Transform


class Lambda(Transform):
    def __init__(self, function, types_to_apply=None, verbose=False):
        super().__init__(verbose=verbose)
        self.function = function
        self.types_to_apply = types_to_apply

    def apply_transform(self, sample):
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            image_type = image_dict['type']
            has_types_list = self.types_to_apply is not None
            if has_types_list and image_type in self.types_to_apply:
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
