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
sample = dataset[0]
#sample = deepcopy(sample_orig)

nT = 100
time_points = [.55, 1.0]
fitpars = np.zeros((6, nT))

fitpars[1, 55:] = -15
#fitpars[dim_modif, :45] = -7.5
#fitpars[dim_modif, 45:] = 7.5

#ov(sample["T1"]["data"][0], sample["T1"]["affine"])

transform = RandomMotionTimeCourseAffines(fitpars=fitpars, time_points=time_points, pct_oversampling=0.30, verbose=True,combine_axis=0)
transformed = transform(sample)
dataset.save_sample(transformed, dict(T1='/home/romain.valabregue/tmp/mot/t1_motion_axis0.nii.gz'))

sample = dataset[0]
transform = RandomMotionTimeCourseAffines(fitpars=fitpars, time_points=time_points, pct_oversampling=0.30, verbose=True,combine_axis=1)
transformed = transform(sample)
dataset.save_sample(transformed, dict(T1='/home/romain.valabregue/tmp/mot/t1_motion_axis1.nii.gz'))

sample = dataset[0]
transform = RandomMotionTimeCourseAffines(fitpars=fitpars, time_points=time_points, pct_oversampling=0.30, verbose=True,combine_axis=2)
transformed = transform(sample)
dataset.save_sample(transformed, dict(T1='/home/romain.valabregue/tmp/mot/t1_motion_axis2.nii.gz'))

#ov(transformed["T1"]["data"][0], sample["T1"]["affine"])
