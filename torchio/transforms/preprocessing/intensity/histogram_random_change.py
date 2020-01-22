import numpy as np
from ....torchio import DATA
from .normalization_transform import NormalizationTransform
from .histogram_standardization import normalize


class HistogramRandomChange(NormalizationTransform):
    """
    Same thins as HistogramStandardization but the landmarks is a random curve
    """
    def __init__(self, masking_method=None,
                 nb_point_ini = 50, nb_smooth=5, verbose=False):
        super().__init__(masking_method=masking_method, verbose=verbose)
        self.nb_point_ini = nb_point_ini
        self.nb_smooth = nb_smooth

    def apply_normalization(self, sample, image_name, mask):
        image_dict = sample[image_name]
        landmarks = self.__get_random_landmarks()
        image_dict[DATA] = normalize(
            image_dict[DATA],
            landmarks,
            mask=mask,
        )

    def __get_random_landmarks(self):
        y = np.squeeze(np.sort(np.random.rand(1, self.nb_point_ini)))
        x1 = np.linspace(0, 1, self.nb_point_ini)
        x2 = np.linspace(0, 1, 100)
        y2 = np.interp(x2, x1, y)
        y2 = self.smooth(y2, self.nb_smooth)
        y2 = np.sort(y2)
        y2 = y2 - np.min(y2)
        y2 = (y2 / np.max(y2)) * 100

        return y2

    def smooth(self, y, box_pts):

        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')

        return y_smooth


    def get_curve_for_sample(yall):
        """
        not use but other alternative to get a random landmarks from a set of landmarks
        :return:
        """
        i = np.random.randint(yall.shape[0])
        y = np.squeeze(yall[i,:])

        i = np.random.randint(y.shape)
        print(i)
        if i<yall.shape[1]/2:
            y = y[i[0]:]
        else:
            y = y[0:i[0]]

        x1 = np.linspace(0, 1, y.shape[0])
        x2 = np.linspace(0, 1, 100)
        y2 = np.interp(x2, x1, y)

        y2 = y2 - np.min(y2)
        y2 = (y2 / np.max(y2)) * 100
        plt.plot(y2)
        return y2
