import platform
import torchio
import torch
import numpy


print('Platform:', platform.platform())  # noqa: T001
print('TorchIO: ', torchio.__version__)  # noqa: T001
print('PyTorch: ', torch.__version__)  # noqa: T001
print('NumPy:   ', numpy.__version__)  # noqa: T001
