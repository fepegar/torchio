from torchio.transforms.augmentation.intensity.random_motion_kspace_time_course import RandomMotionTimeCourseAffines
from torchio import Image, ImagesDataset, transforms, INTENSITY, LABEL
from torchvision.transforms import Compose
import numpy as np
from nibabel.viewers import OrthoSlicer3D as ov

np.random.seed(12)

out_dir = '/data/ghiles/'

subject = [[
    Image('T1', '/data/romain/HCPdata/suj_100307/T1w_1mm.nii.gz', INTENSITY),
    Image('mask', '/data/romain/HCPdata/suj_100307/brain_mT1w_1mm.nii', LABEL)
     ]]
subjects_list = [subject]
dataset = ImagesDataset(subject)
sample = dataset[0]

nT = 200
time_points = [.5, 1.0]
dim_modif = 4
fitpars = np.zeros((6, nT))
fitpars[dim_modif, :100] = 12

ov(sample["T1"]["data"][0], sample["T1"]["affine"])

transform = RandomMotionTimeCourseAffines(fitpars=fitpars, time_points=time_points, pct_oversampling=0.30, verbose=True)
transformed = transform(sample)


ov(transformed["T1"]["data"][0], sample["T1"]["affine"])
