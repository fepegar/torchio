# flake8: noqa

import re
import sys
import platform
import torchio
import torch
import numpy
import SimpleITK as sitk


sitk_version = re.findall('SimpleITK Version: (.*?)\n', str(sitk.Version()))[0]

print('Platform:  ', platform.platform())
print('TorchIO:   ', torchio.__version__)
print('PyTorch:   ', torch.__version__)
print('SimpleITK: ', sitk_version)
print('NumPy:     ', numpy.__version__)
print('Python:    ', sys.version)
