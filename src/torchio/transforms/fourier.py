import numpy as np
import torch


class FourierTransform:
    @staticmethod
    def fourier_transform(tensor: torch.Tensor) -> torch.Tensor:
        try:
            import torch.fft

            transformed = torch.fft.fftn(tensor)
            fshift = torch.fft.fftshift(transformed)
            return fshift
        except (ModuleNotFoundError, AttributeError):
            import torch

            transformed = np.fft.fftn(tensor)
            fshift = np.fft.fftshift(transformed)
            return torch.from_numpy(fshift)

    @staticmethod
    def inv_fourier_transform(tensor: torch.Tensor) -> torch.Tensor:
        try:
            import torch.fft

            f_ishift = torch.fft.ifftshift(tensor)
            img_back = torch.fft.ifftn(f_ishift)
            return img_back
        except (ModuleNotFoundError, AttributeError):
            import torch

            f_ishift = np.fft.ifftshift(tensor)
            img_back = np.fft.ifftn(f_ishift)
            return torch.from_numpy(img_back)
