from pprint import pprint
from torchio import Image, ImagesDataset, transforms, INTENSITY, LABEL
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
import numpy as np
import os, math, sys

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
    #fp[1,:] = y
    return fp

def corrupt_data_both( x0, sigma= 5, amplitude=20, method='gauss'):
    fp1 = corrupt_data(x0, sigma, amplitude=amplitude, method='gauss')
    fp2 = corrupt_data(30, 2, amplitude=-amplitude, method='step')
    fp = fp1 + fp2
    return fp


dico_params = {    "fitpars": None,  "verbose": True, "displacement_shift":0 , "oversampling_pct":0,
                   "correct_motion":True, "nufft":True}

do_plot=False
z_slice = [152, 182, 218, 256, 512]
z_slice = [512, 256, 182]
z_slice = [512]

dirpath = ['/data/romain/data_exemple/motion_bug']

fp = corrupt_data(80, sigma=1,method='step',amplitude=20)
fp = np.zeros([6,200]); fp[3, :] = 20

dico_params['fitpars'] = fp

for zs in z_slice:
    t = RandomMotionFromTimeCourse(**dico_params)
    dataset = ImagesDataset(suj, transform=Compose((CenterCropOrPad(target_shape=(182, 218, zs)), t)))
    sample = dataset[0]

    fout = dirpath[0] + '/mot_step_rot_xN{}'.format(zs)
    dataset.save_sample(sample, dict(image=fout+'.nii'))

    if do_plot:
        #check the fitpars
        fit_pars = t.fitpars
        fig = plt.figure('fitpars_{}'.format(zs)); plt.plot(fit_pars.T)
        fit_pars = t.fitpars_interp
        fig = plt.figure('fit_interp_{}'.format(zs)); plt.plot(fit_pars.reshape(6,-1).T)


sys.exit(1)

#3D case
#def rotate_coordinates():
import math
from torchio.transforms.augmentation.intensity.random_motion_from_time_course import create_rotation_matrix_3d

if True:
    rotations = t.rotations

    center = [math.ceil((x - 1) / 2) for x in t.im_shape]

    [i1, i2, i3] = np.meshgrid(np.arange(t.im_shape[0]) - center[0],
                               np.arange(t.im_shape[1]) - center[1],
                               np.arange(t.im_shape[2]) - center[2], indexing='ij')

    grid_coordinates = np.array([i1.T.flatten('F'), i2.T.flatten('F'), i3.T.flatten('F')])

    #print('rotation size is {}'.format(t.rotations.shape))

    rotations = rotations.reshape([3] + t.im_shape)
    ix = (len(t.im_shape) + 1) * [slice(None)]
    ix[t.frequency_encoding_dim + 1] = 0  # dont need to rotate along freq encoding

    rotations = rotations[tuple(ix)].reshape(3, -1)
    rotation_matrices = np.apply_along_axis(create_rotation_matrix_3d, axis=0, arr=rotations).transpose([-1, 0, 1])
    #rotation_matrices = np.apply_along_axis(create_rotation_matrix_3d, axis=0, arr=rotations).transpose([-1, 1, 0])
    rotation_matrices = rotation_matrices.reshape(t.phase_encoding_shape + [3, 3])
    rotation_matrices = np.expand_dims(rotation_matrices, t.frequency_encoding_dim)

    rotation_matrices = np.tile(rotation_matrices,
                                reps=([t.im_shape[ t.frequency_encoding_dim] if i == t.frequency_encoding_dim else 1
                                       for i in range(5)]))  # tile in freq encoding dimension

    #bug fix same order F as for grid_coordinates where it will be multiply to
    rotation_matrices = rotation_matrices.reshape([-1, 3, 3], order='F')

    # tile grid coordinates for vectorizing computation
    grid_coordinates_tiled = np.tile(grid_coordinates, [3, 1])
    grid_coordinates_tiled = grid_coordinates_tiled.reshape([3, -1], order='F').T
    rotation_matrices = rotation_matrices.reshape([-1, 3])  # reshape for matrix multiplication, so no order F

    new_grid_coords = (rotation_matrices * grid_coordinates_tiled).sum(axis=1)

    # reshape new grid coords back to 3 x nvoxels
    new_grid_coords = new_grid_coords.reshape([3, -1], order='F')
    # new_grid_coords = new_grid_coords.reshape([3] + self.im_shape, order='C')
    # new_grid_coords = new_grid_coords.reshape([3, -1] , order='F')


    # scale data between -pi and pi
    max_vals = [abs(x) for x in grid_coordinates[:, 0]]
    max_vals_test = [np.max(np.abs(grid_coordinates[i, :])) for i in range(0, 3)]  # rrr

    [np.max(np.abs(grid_coordinates[i, :])) for i in range(0, 3)]

    if not (max_vals_test == max_vals) : print('RRR which on ?')
    #change in volume : max_vals = [np.max(np.abs(new_grid_coords[i, :])) for i in range(0,3) ] #rrr

    new_grid_coordinates_scaled = [(new_grid_coords[i, :] / max_vals[i]) * math.pi for i in
                                   [2,1,0]]
                                #range(new_grid_coords.shape[0])]
    new_grid_coordinates_scaled = [np.asfortranarray(i) for i in new_grid_coordinates_scaled]
    #rrr why already flat ... ?


    #return new_grid_coordinates_scaled, [grid_coordinates, new_grid_coords]

pts = np.array(grid_coordinates[:,-1])
new_grid_coords[:,-1]
rot=rotation_matrices[-1,:]
pts_rot = np.matmul(rot, pts)


#1D case no difference
rotations = np.tile(np.array([[np.radians(20), 0, 0]]).T,[1,4])
#rotations = np.tile(np.array([[0, np.radians(45), 0]]).T,[1,4])
grid_coordinates = np.tile(np.array([1, 2, 3, 4]), [3, 1])
pts = np.tile(np.array([1, 2, 3, 4]), [3, 1])

print('rot {} pts {}'.format(rotations.shape, grid_coordinates.shape))

grid_coordinates_tiled = np.tile(grid_coordinates, [3, 1])
grid_coordinates_tiled.shape
grid_coordinates_tiled = grid_coordinates_tiled.reshape([3, -1], order='F').T
grid_coordinates_tiled.shape

rotation_matrices = np.apply_along_axis(create_rotation_matrix_3d, axis=0, arr=rotations).transpose([-1, 0, 1])
rot_mat = np.apply_along_axis(create_rotation_matrix_3d, axis=0, arr=rotations).transpose([-1, 0, 1])

rotation_matrices = rotation_matrices.reshape([-1, 3])
new_grid_coords = (rotation_matrices * grid_coordinates_tiled).sum(axis=1)
new_grid_coords = new_grid_coords.reshape([3, -1], order='F')


#version matriceil
rr = rot_mat[0]
pts_rot = np.matmul(rr, pts)
#print(rr)
print(pts_rot)
pts_rot - new_grid_coords

