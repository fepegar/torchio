from typing import Any
from typing import Dict
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .subject import Subject


class SubjectsLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            collate_fn=self._collate,
            **kwargs,
        )

    @staticmethod
    def _collate(subjects: List[Subject]) -> Dict[str, Any]:
        first_subject = subjects[0]
        batch_dict = {}
        for key in first_subject.keys():
            collated_value = _stack([subject[key] for subject in subjects])
            batch_dict[key] = collated_value
        return batch_dict


def _stack(x):
    """Determine the type of the input and stack it accordingly.

    Args:
        x: List of elements to stack.
    Returns:
        Stacked elements, as either a torch.Tensor, np.ndarray, dict or list.
    """
    first_element = x[0]
    if isinstance(first_element, torch.Tensor):
        return torch.stack(x, dim=0)
    elif isinstance(first_element, np.ndarray):
        return np.stack(x, axis=0)
    elif isinstance(first_element, dict):
        # Assume that all elements have the same keys
        collated_dict = {}
        for key in first_element.keys():
            collated_dict[key] = _stack([element[key] for element in x])
        return collated_dict
    else:
        return x
