from abc import ABC

from tensordict.tensordict import TensorDict


class BaseContainer(TensorDict, ABC):
    def __iter__(self):
        return iter(self.keys())
