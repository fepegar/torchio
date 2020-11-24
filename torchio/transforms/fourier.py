import numpy as np


class FourierTransform:

    @staticmethod
    def fourier_transform(array: np.ndarray) -> np.ndarray:
        transformed = np.fft.fftn(array)
        fshift = np.fft.fftshift(transformed)
        return fshift

    @staticmethod
    def inv_fourier_transform(fshift: np.ndarray) -> np.ndarray:
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifftn(f_ishift)
        return img_back
