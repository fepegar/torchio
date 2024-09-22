import platform
import re
import sys

import numpy
import SimpleITK as sitk
import torch

import torchio as tio

sitk_version = re.findall('SimpleITK Version: (.*?)\n', str(sitk.Version()))[0]

print('Platform:  ', platform.platform())
print('TorchIO:   ', tio.__version__)
print('PyTorch:   ', torch.__version__)
print('SimpleITK: ', sitk_version)
print('NumPy:     ', numpy.__version__)
print('Python:    ', sys.version)
