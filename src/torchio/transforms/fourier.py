import torch
import numpy as np


class FourierTransform:

    @staticmethod
    def fourier_transform(tensor: torch.Tensor) -> torch.Tensor:
        try:
            import torch.fft
            return torch.fft.fftn(tensor)
        except ModuleNotFoundError:
            import torch
            transformed = np.fft.fftn(tensor)
            fshift = np.fft.fftshift(transformed)
            return torch.from_numpy(fshift)

    @staticmethod
    def inv_fourier_transform(tensor: torch.Tensor) -> torch.Tensor:
        try:
            import torch.fft
            return torch.fft.ifftn(tensor)
        except ModuleNotFoundError:
            import torch
            f_ishift = np.fft.ifftshift(tensor)
            img_back = np.fft.ifftn(f_ishift)
            return torch.from_numpy(img_back)
