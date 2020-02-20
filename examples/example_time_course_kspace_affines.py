from torchio.transforms.augmentation.intensity.random_motion_kspace_time_course import RandomMotionTimeCourseAffines
from torchio import Image, ImagesDataset, transforms, INTENSITY, LABEL
from torchvision.transforms import Compose
import numpy as np
from nibabel.viewers import OrthoSlicer3D as ov
from copy import deepcopy

np.random.seed(12)

out_dir = '/data/ghiles/'

subject = [[
    Image('T1', '/data/romain/HCPdata/suj_100307/T1w_1mm.nii.gz', INTENSITY),
    Image('mask', '/data/romain/HCPdata/suj_100307/brain_mT1w_1mm.nii', LABEL)
     ]]
subjects_list = [subject]
dataset = ImagesDataset(subject)
sample_orig = dataset[0]
#sample = deepcopy(sample_orig)
sample = sample_orig

nT = 100
time_points = [.58, 1.0]
dim_modif = 1
fitpars = np.zeros((6, nT))

fitpars[0, 58:] = -94
fitpars[1, 58:] = -5
fitpars[2, 58:] = -88
#fitpars[dim_modif, :45] = -7.5
#fitpars[dim_modif, 45:] = 7.5

#ov(sample["T1"]["data"][0], sample["T1"]["affine"])

transform = RandomMotionTimeCourseAffines(fitpars=fitpars, time_points=time_points, pct_oversampling=0.30, verbose=True)
#transform = RandomMotionTimeCourseAffines(fitpars=fitpars, time_points=time_points, pct_oversampling=0.30, verbose=True,combine_axis=0)

transformed = transform(sample)

dataset.save_sample(transformed, dict(T1='/home/romain.valabregue/tmp/mot/t1_motion.nii.gz'))

#ov(transformed["T1"]["data"][0], sample["T1"]["affine"])
