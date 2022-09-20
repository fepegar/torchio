import numpy
import torch


class FourierTransform:

    @staticmethod
    def fourier_transform(tensor: torch.Tensor) -> torch.Tensor:
        transformed = numpy.fft.fftn(tensor)
        fshift = numpy.fft.fftshift(transformed)
        return torch.from_numpy(fshift)

    @staticmethod
    def inv_fourier_transform(tensor: torch.Tensor) -> torch.Tensor:
        f_ishift = numpy.fft.ifftshift(tensor)
        img_back = numpy.fft.ifftn(f_ishift)
        return torch.from_numpy(img_back)
