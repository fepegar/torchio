# flake8: noqa

import sys
import platform
import torchio
import torch
import numpy

print('Platform:', platform.platform())
print('TorchIO: ', torchio.__version__)
print('PyTorch: ', torch.__version__)
print('NumPy:   ', numpy.__version__)
print('Python:  ', sys.version)
