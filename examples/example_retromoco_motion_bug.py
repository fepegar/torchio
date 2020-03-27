from pprint import pprint
from torchio import Image, ImagesDataset, transforms, INTENSITY, LABEL
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
import numpy as np
import os

from torchio.transforms import RandomMotionFromTimeCourse, RandomAffine, CenterCropOrPad
from copy import deepcopy
from nibabel.viewers import OrthoSlicer3D as ov
from torchvision.transforms import Compose

"""
Comparing result with retromocoToolbox
"""
import pandas as pd

from torchio.transforms import Interpolation
#suj = [[ Image('T1', '/data/romain/HCPdata/suj_150423/mT1w_1mm.nii', INTENSITY), ]]
suj = [[ Image('image', '/data/romain/data_exemple/suj_274542/mask_brain.nii', INTENSITY), ]]
#suj = [[Image('image', '/data/romain/HCPdata/suj_150423/T1w_1mm.nii.gz', INTENSITY), ]]

def corrupt_data( x0, sigma= 5, amplitude=20, method='gauss'):
    fp = np.zeros((6, 200))
    x = np.arange(0,200)
    if method=='gauss':
        y = np.exp(-(x - x0) ** 2 / float(2 * sigma ** 2))*amplitude
    elif method == 'step':
        if x0<100:
            y = np.hstack((np.zeros((1,(x0-sigma))),
                            np.linspace(0,amplitude,2*sigma+1).reshape(1,-1),
                            np.ones((1,((200-x0)-sigma-1)))*amplitude ))
        else:
            y = np.hstack((np.zeros((1,(x0-sigma))),
                            np.linspace(0,-amplitude,2*sigma+1).reshape(1,-1),
                            np.ones((1,((200-x0)-sigma-1)))*-amplitude ))
    fp[3,:] = y
    fp[1,:] = y
    return fp

def corrupt_data_both( x0, sigma= 5, amplitude=20, method='gauss'):
    fp1 = corrupt_data(x0, sigma, amplitude=amplitude, method='gauss')
    fp2 = corrupt_data(30, 2, amplitude=-amplitude, method='step')
    fp = fp1 + fp2
    return fp


dico_params = {    "fitpars": None,  "verbose": True, "displacement_shift":1 , "oversampling_pct":0, "correct_motion":True}

do_plot=False
z_slice = [152, 182, 218, 256, 512]
#z_slice = [152]

dirpath = ['/data/romain/data_exemple/motion_bug']

fp = corrupt_data(80, sigma=1,method='step',amplitude=20)
dico_params['fitpars'] = fp

for zs in z_slice:
    t = RandomMotionFromTimeCourse(**dico_params)
    dataset = ImagesDataset(suj, transform=Compose((CenterCropOrPad(target_shape=(182, 218, zs)), t)))
    sample = dataset[0]

    fout = dirpath[0] + '/mot_step_z{}'.format(zs)
    dataset.save_sample(sample, dict(image=fout+'.nii'))

    if do_plot:
        #check the fitpars
        fit_pars = t.fitpars
        fig = plt.figure('fitpars_{}'.format(zs)); plt.plot(fit_pars.T)
        fit_pars = t.fitpars_interp
        fig = plt.figure('fit_interp_{}'.format(zs)); plt.plot(fit_pars.reshape(6,-1).T)

