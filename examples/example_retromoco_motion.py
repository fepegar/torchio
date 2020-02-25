from pprint import pprint
from torchio import Image, ImagesDataset, transforms, INTENSITY, LABEL
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
import numpy as np

from torchio.transforms import RandomMotionFromTimeCourse, RandomAffine
from copy import deepcopy
from nibabel.viewers import OrthoSlicer3D as ov

"""
Comparing result with retromocoToolbox
"""
from utils_file import gfile, get_parent_path

from torchio.transforms import Interpolation
suj = [[ Image('T1', '/data/romain/HCPdata/suj_274542/mT1w_1mm.nii', INTENSITY), ]]

rp_files = gfile('/data/romain/HCPdata/suj_274542/Motion_ms','^rp')

rpf = rp_files[10]
for rpf in rp_files:
    dirpath,name = get_parent_path([rpf])
    fout = dirpath[0] + '/check/'+name[0][3:-4] + '.nii'

    t = RandomMotionFromTimeCourse(fitpars=rpf, nufft=True, oversampling_pct=0, keep_original=True, verbose=True)
    dataset = ImagesDataset(suj, transform=t)
    sample = dataset[0]

    dataset.save_sample(sample, dict(T1=fout))

fit_pars = sample['T1']['fit_pars']
plt.figure; plt.plot(fit_pars[3:].T)
plt.figure; plt.plot(fit_pars.T)








#test transforms
from torchio.transforms import RandomSpike
t = RandomSpike(num_spikes_range=(5,10), intensity_range=(0.1,0.2))
dataset = ImagesDataset(suj, transform=t)

for i in range(1,10):
    sample = dataset[0]
    fout='/tmp/toto{}.nii'.format(i)
    dataset.save_sample(sample, dict(T1=fout))





out_dir = '/data/ghiles/motion_simulation/tests/'

def corrupt_data(data, percentage):
    n_pts_to_corrupt = int(round(percentage * len(data)))
    #pts_to_corrupt = np.random.choice(range(len(data)), n_pts_to_corrupt, replace=False)
    # MotionSimTransformRetroMocoBox.perlinNoise1D(npts=n_pts_to_corrupt,
    #                                        weights=np.random.uniform(low=1.0, high=2)) - .5
    #to avoid global displacement let the center to zero
    if percentage>0.5:
        data[n_pts_to_corrupt:] = 15
    else:
        data[:n_pts_to_corrupt] = 15

    return data


dico_params = {
    "maxDisp": 0,
    "maxRot": 0,
    "tr": 2.3,
    "es": 4e-3,
    "nT": 200,
    "noiseBasePars": 0,
    "swallowFrequency": 0,
    "swallowMagnitude": 0,
    "suddenFrequency": 0,
    "suddenMagnitude": 0,
    "displacement_shift": 0,
    "freq_encoding_dim": [1],
    "oversampling_pct": 0.3,
    "nufft": True,
    "verbose": True,
}


np.random.seed(12)
suj = [[
    Image('T1', '/data/romain/HCPdata/suj_100307/T1w_1mm.nii.gz', INTENSITY),
    Image('mask', '/data/romain/HCPdata/suj_100307/brain_mT1w_1mm.nii', LABEL)
     ]]

corrupt_pct = [.25, .45, .55, .75]
corrupt_pct = [.45]
transformation_names = ["translation1", "translation2", "translation3", "rotation1", "rotation2", "rotation3"]
fpars_list = dict()
dim_loop = [0, 1, 2]
for dd in dim_loop:
    for pct_corr in corrupt_pct:
        fpars_list[pct_corr] = dict()
        for dim, name in enumerate(transformation_names):
            fpars_handmade = np.zeros((6, dico_params['nT']))
            fpars_handmade[dim] = corrupt_data(fpars_handmade[dim], pct_corr)
            #fpars_handmade[3:] = np.radians(fpars_handmade[3:])
            fpars_list[pct_corr][name] = fpars_handmade
            dico_params["fitpars"] = fpars_handmade
            #dico_params["freq_encoding_dim"] = [dim % 3]
            dico_params["freq_encoding_dim"] = [dd]

            t = RandomMotionFromTimeCourse(**dico_params)
            transforms = Compose([t])
            dataset = ImagesDataset(suj, transform=transforms)
            sample = dataset[0]
    #        dataset.save_sample(sample, dict(T1='/data/romain/data_exemple/motion/begin_{}_{}_freq{}_Center{}.nii'.format(
    #            name, pct_corr,dico_params["freq_encoding_dim"][0],dico_params["displacement_shift"])))
            dataset.save_sample(sample, dict(T1='/data/romain/data_exemple/motion/noorderF_{}_{}_freq{}.nii'.format(
                name, pct_corr,dico_params["freq_encoding_dim"][0])))
            print("Saved {}_{}".format(name, pct_corr))



t = RandomMotionFromTimeCourse(**dico_params)
transforms = Compose([t])
dataset = ImagesDataset(suj, transform=transforms)
sample = dataset[0]

rots = t.rotations.reshape((3, 182, 218, 182))
translats = t.translations.reshape((3, 182, 218, 182))






# TESTING AFFINE GRIG from pytorch
from torchio.transforms.augmentation.intensity.random_motion_from_time_course import create_rotation_matrix_3d
#import sys
#sys.path.append('/data/romain/toolbox_python/romain/cnnQC/')
#from utils import reslice_to_ref
import nibabel.processing as nbp
import nibabel as nib
import torch.nn.functional as F
import torch
sample = dataset[0]
ii, affine = sample['T1']['data'], sample['T1']['affine']

rot = np.deg2rad([0,10,20])
scale = [1, 1.2, 1/1.2 ]
trans = [-30, 30, 0]
image_size = np.array([ii[0].size()])
trans_torch = np.array(trans)/(image_size/2)
mr = create_rotation_matrix_3d(rot)
ms = np.diag(scale)
center = np.ceil(image_size/2)
center = center.T -  mr@center.T
center_mat=np.zeros([4,4])
center_mat[0:3,3] = center[0:3].T
maff = np.hstack((ms @ mr,np.expand_dims(trans,0).T))
maff_torch = np.hstack((ms @ mr,trans_torch.T))
maff = np.vstack((maff,[0,0,0,1]))

nib_fin  = nib.Nifti1Image(ii.numpy()[0], affine)
new_aff = affine @ np.linalg.inv(maff+center_mat) #new_aff = maff @ affine # other way round  new_aff = affine@maff
nib_fin.affine[:] = new_aff[:]
fout = nbp.resample_from_to(nib_fin, (nib_fin.shape, affine), cval=-1) #fout = nbp.resample_from_to(nib_fin, (nib_fin.shape, new_aff), cval=-1)
ov(fout.get_fdata())
#it gives almost the same, just the scalling is shifted with nibabel (whereas it is centred with torch

mafft = maff_torch[np.newaxis,:]
mafft = torch.from_numpy(mafft)

x = ii.permute(0,3,2,1).unsqueeze(0)
grid = F.affine_grid(mafft, x.shape, align_corners=False).float()
x = F.grid_sample(x, grid, align_corners=False)
xx = x[0,0].numpy().transpose(2,1,0)
ov(xx)

# make the inverse transform
xx=torch.zeros(4,4); xx[3,3]=1
xx[0:3,0:4] = mafft[0]
imaf = xx.inverse()
imaf = imaf[0:3,0:4].unsqueeze(0)

grid = F.affine_grid(imaf, x.shape, align_corners=False).float()
x = F.grid_sample(x, grid, align_corners=False)
xx = x[0,0].numpy().transpose(2,1,0)
ov(xx)
