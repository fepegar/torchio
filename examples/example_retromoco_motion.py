from pprint import pprint
from torchio import Image, ImagesDataset, transforms, INTENSITY, LABEL, Subject
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
from utils_file import gfile, get_parent_path
import pandas as pd
from doit_train import do_training

def corrupt_data( x0, sigma= 5, amplitude=20, method='gauss', mvt_axes=[1] ):
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
    elif method == 'sin':
        fp = np.zeros((6, 182*218))
        x = np.arange(0,182*218)
        y = np.sin(x/x0 * 2 * np.pi)
        #plt.plot(x,y)

    for xx in mvt_axes:
        fp[xx,:] = y
    return fp

def corrupt_data_both( x0, sigma= 5, amplitude=20, method='gauss'):
    fp1 = corrupt_data(x0, sigma, amplitude=amplitude, method='gauss')
    fp2 = corrupt_data(30, 2, amplitude=-amplitude, method='step')
    fp = fp1 + fp2
    return fp


dt = do_training('/tmp', 'toto')
res, extra_info = pd.DataFrame(), dict()

suj_type='suj'
if suj_type=='suj':
    suj = [ Subject(image=Image('/data/romain/data_exemple/suj_150423/mT1w_1mm.nii', INTENSITY)), ]
else:
    suj = [ Subject(image=Image('/data/romain/data_exemple/suj_150423/mask_brain.nii', INTENSITY)), ]

dico_params = {    "fitpars": None,  "verbose": True, "displacement_shift":1 , "oversampling_pct":0,
                    'keep_original':True, 'compare_to_original':True, "correct_motion":False,
                   'freq_encoding_dim':[2]}

disp_str =  'no_shift' # 'center_zero'  'no_shift' #'center_TF'
disp_str_list = ['center_zero', 'no_shift', 'center_TF']

x0=np.hstack((np.arange(90,102,2),np.arange(101,105,1))) #x0=[100]
x0=np.hstack((np.arange(90,102,2))) #x0=[100]
x0 = np.hstack((np.array([20, 50, 80, 90, 100])))
x0 = np.array([218*2, 218*(2+1/3), 218*4])
x0 = np.array([ 197, 218, 218*2, 218*(2+11/7), 218*4])
x0 = np.array([ 145, 152, 152*2, 152*(2+11/7), 152*4])
x0s = [ np.array([ 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 96, 97, 98, 99, 100]) ,
        np.array([ 10, 20, 30, 40, 50, 60, 70, 80, 88, 90, 92, 94, 96, 98, 100]) ]
mvt_types=['step', 'gauss']
mvt_type, x0 =mvt_types[1], x0s[0]

mvt_axe_str_list = ['transX', 'transY','transZ', 'rotX', 'rotY', 'rotZ']
mvt_axes = [1]
mvt_axe_str = mvt_axe_str_list[mvt_axes[0]]
out_path = '/data/romain/data_exemple/motion_gaussX_sigma2_phaseX'
if not os.path.exists(out_path): os.mkdir(out_path)
#plt.ioff()
#for mvt_type, x0 in zip(mvt_types, x0s):
for disp_str in disp_str_list:
    for s in [2]: #[2, 5, 10, 20]: #[1, 2, 3,  5, 7, 10, 12 , 15, 20 ] : # [2,4,6] : #[1, 3 , 5 , 8, 10 , 12, 15, 20 , 25 ]:
        for xx in x0:
            if disp_str == 'center_TF': dico_params['displacement_shift'] = 2
            if disp_str == 'center_zero': dico_params['displacement_shift'] = 1
            if disp_str == 'no_shift': dico_params['displacement_shift'] = 0

            fp = corrupt_data(xx, sigma=s, method=mvt_type, amplitude=20, mvt_axes=mvt_axes)
            dico_params['fitpars'] = fp
            dico_params['nT'] = fp.shape[1]
            t =  RandomMotionFromTimeCourse(**dico_params)
            #dataset = ImagesDataset(suj, transform=Compose((CenterCropOrPad(target_shape=(182, 218, 152)), t)))
            dataset = ImagesDataset(suj, transform=t)
            sample = dataset[0]
            fout = out_path + '/{}_{}_{}_freq{}_{}'.format(suj_type, mvt_axe_str, mvt_type, xx, disp_str)
            fit_pars = t.fitpars
            fig = plt.figure()
            plt.plot(fit_pars.T)
            plt.savefig(fout+'.png')
            plt.close(fig)
            dataset.save_sample(sample, dict(image=fout+'.nii'))
            extra_info['x0'], extra_info['mvt_type'], extra_info['mvt_axe']= xx, mvt_type, mvt_axe_str
            extra_info['shift'] = disp_str
            res = dt.add_motion_info(sample, res, extra_info=extra_info)
fres = out_path+'/res_metrics_{}_{}.csv'.format(mvt_axe_str, disp_str)
res.to_csv(fres)

res = pd.read_csv('/data/romain/data_exemple/motion_gaussX/res_metrics_transX_center_TF.csv')
#res = pd.read_csv('/data/romain/data_exemple/motion_gaussX_sigma2/res_metrics_transX_center_TF.csv')
res = pd.read_csv('/data/romain/data_exemple/motion_stepX/res_metrics_transX_step.csv')
isel = [range(0,15), range(15,30), range(30,45)]
for ii in isel:
    plt.figure('ssim')
    plt.plot( res.loc[ii,'x0'], res.loc[ii,'ssim'])
    plt.figure('displacement')
    plt.plot(res.loc[ii, 'x0'], res.loc[ii, 'mean_DispP_iterp']) #mean_DispP_iterp  rmse_Disp_iterp

plt.figure('ssim')
plt.legend(disp_str_list)
plt.grid(); plt.ylabel('ssim'); plt.xlabel('')
plt.figure('displacement')
plt.legend(disp_str_list)
plt.grid(); plt.ylabel('displacement'); plt.xlabel('')


fitpars  =t.fitpars_interp
plt.plot(fitpars[1].reshape(-1)) #order C par defaut : with the last axis index changing fastest -> display is correct
ff=np.tile(np.expand_dims(fitpars,1),(1,182,1,1))
#ff=np.moveaxis(ff,2,3)
#plt.plot(ff[1].reshape(-1,order='F'))

fitpars_interp =ff
dd = ImagesDataset(suj, transform=CenterCropOrPad(size=(182, 218,152)) ); sorig = dd[0]
original_image = sorig['T1']['data'][0]

#pour amplitude de 40 presque
#Removing [ 0.        -2.8949889  0.         0.         0.         0.       ] OR [0.         2.51842243 0.        -> first 5.41
#?? [ 0.         -3.23879857  0.          0.          0.          0.        ] OR [0.         2.17461276 0.         0.         0.         0.        ]



dataset = ImagesDataset(suj)
so=dataset[0]
image = so['T1']['data'][0]
tfi = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)
tfi_sum = np.abs(np.sum(tfi,axis=0)); #tfi_sum = np.sum(np.abs(tfi),axis=0)
sum_intensity, sum_intensity_abs = np.zeros((tfi.shape[2])), np.zeros((tfi.shape[2]))
sum_intensity, sum_intensity_abs = np.zeros((tfi.shape[1],tfi.shape[2])), np.zeros((tfi.shape[1],tfi.shape[2]))
#for z in range(0,tfi.shape[2]):
for y in range(0, tfi.shape[1]):
    for z in range(0, tfi.shape[2]):
        ttf = np.zeros(tfi.shape,dtype=complex)
        ttf[:,y,z] = tfi[:,y,z]
        ifft = np.fft.ifftshift(np.fft.ifftn(ttf))
        sum_intensity[y,z] = np.abs(np.sum(ifft))
        sum_intensity_abs[y,z] = np.sum(np.abs(ifft))
sum_intensity_abs = np.load('/data/romain/data_exemple/suj_274542/intensity_fft_mask.npz.npy')

for s in [1,2, 3, 4, 5 , 8, 10 , 12, 15, 20 , 2500 ]:
    fp = corrupt_data(50, sigma=s, method='gauss')
    dico_params['fitpars'] = fp
    t = RandomMotionFromTimeCourse(**dico_params)

    t._calc_dimensions(sample['T1']['data'][0].shape)
    fitpars_interp = t._interpolate_space_timing(t.fitpars)
    fitpars_interp = np.tile(fitpars_interp,[1,182,1,1])
    trans = fitpars_interp[1,0,:]
    #plt.figure(); plt.plot(trans.reshape(-1))
    print(np.sum(trans*sum_intensity_abs)/np.sum(sum_intensity_abs))

fp = corrupt_data(109,5,amplitude=40 )
ffp = np.expand_dims(np.expand_dims(fp,1),3)
ff = np.tile(ffp, [1, 182, 1, 152])


#testing with smal fitpars if mean of rot is the same as mean of affine
ff=fitpars=np.abs(t.fitpars)
ss = np.ones(ff.shape)
to_substract = np.zeros(6)
for i in range(0, 6):
    ffi = ff[i].reshape(-1, order='F')
    ssi = ss[i].reshape(-1, order='C')
    # mean over all kspace
    to_substract[i] = np.sum(ffi * ssi) / np.sum(ssi)

fitpars = fitpars - np.tile(to_substract[...,np.newaxis],[1,200])

from torchio.transforms.augmentation.intensity.random_motion_from_time_course import create_rotation_matrix_3d

affine = np.identity(4)
rot = np.radians(fitpars[3:])
rotation_matrices = np.apply_along_axis(create_rotation_matrix_3d, axis=0, arr=rot).transpose([-1, 0, 1])
tt = fitpars[0:3, :].transpose([1, 0])
affs = np.tile(affine, [fitpars.shape[1], 1, 1])
affs[:,0:3,0:3] = rotation_matrices
affs[:, 0:3, 3] = tt

from scipy.linalg import logm, expm
weights, matrices = ss[0], affs

logs = [w * logm(A) for (w, A) in zip(weights, matrices)]
logs = np.array(logs)
logs_sum = logs.sum(axis=0)
expm(logs_sum/np.sum(weights, axis=0) )
#a 10-2 pres c'est bien l'identite !

rp_files = gfile('/data/romain/HCPdata/suj_274542/Motion_ms','^rp')
rp_files = gfile('/data/romain/HCPdata/suj_274542/mot_separate','^rp')

rpf = rp_files[10]
res = pd.DataFrame()
for rpf in rp_files:
    dirpath,name = get_parent_path([rpf])
    fout = dirpath[0] + '/check/'+name[0][3:-4] + '.nii'

    t = RandomMotionFromTimeCourse(fitpars=rpf, nufft=True, oversampling_pct=0, keep_original=True, verbose=True)
    dataset = ImagesDataset(suj, transform=t)
    sample = dataset[0]
    dicm = sample['T1']['metrics']
    dicm['fname'] = fout
    res = res.append(dicm, ignore_index=True)
    dataset.save_sample(sample, dict(T1=fout))

fit_pars = sample['T1']['fit_pars']
plt.figure; plt.plot(fit_pars[3:].T)
plt.figure; plt.plot(fit_pars.T)


dic_no_mot ={ "noiseBasePars": (5, 20, 0),"swallowFrequency": (0, 1, 1), "suddenFrequency": (0, 1, 1),
              "oversampling_pct":0.3, "nufft":True , "keep_original": True}
t = RandomMotionFromTimeCourse(**dic_no_mot)
dataset = ImagesDataset(suj, transform=t)
sample = dataset[0]


dico_params = {"maxDisp": (1, 6),  "maxRot": (1, 6),    "noiseBasePars": (5, 20, 0),
               "swallowFrequency": (2, 6, 0),  "swallowMagnitude": (1, 6),
               "suddenFrequency": (1, 2, 1),  "suddenMagnitude": (6, 6),
               "verbose": True, "keep_original": True, "compare_to_original": True}

dico_params = {"maxDisp": (1, 6),  "maxRot": (1, 6),    "noiseBasePars": (5, 20, 0.8),
               "swallowFrequency": (2, 6, 0.5),  "swallowMagnitude": (1, 6),
               "suddenFrequency": (2, 6, 0.5),  "suddenMagnitude": (1, 6),
               "verbose": True, "keep_original": True, "compare_to_original": True, "oversampling_pct":0,
               "preserve_center_pct":0.01}
dico_params = {"maxDisp": (1, 6),  "maxRot": (1, 6),    "noiseBasePars": (5, 20, 0.8),
               "swallowFrequency": (2, 6, 0.5),  "swallowMagnitude": (3, 6),
               "suddenFrequency": (2, 6, 0.5),  "suddenMagnitude": (3, 6),
               "verbose": False, "keep_original": True, "proba_to_augment": 1,
               "preserve_center_pct":0.1, "keep_original": True, "compare_to_original": True,
               "oversampling_pct":0, "correct_motion":True}

np.random.seed(12)
t = RandomMotionFromTimeCourse(**dico_params)
dataset = ImagesDataset(suj, transform=t)

res = pd.DataFrame()
dirpath = ['/data/romain/data_exemple/motion_random_preserve01/'];
if not os.path.isdir(dirpath[0]): os.mkdir(dirpath[0])
plt.ioff()
for i in range(500):
    sample = dataset[0]
    dicm = sample['T1']['metrics']
    dics = sample['T1']['simu_param']
    fout = dirpath[0]  +'mot_TF_fit_par_sim{}'.format(np.floor(dicm['ssim']*10000))
    dicm['fname'] = fout
    dicm.update(dics)
    fit_pars = t.fitpars
    np.savetxt(fout+'.csv', fit_pars, delimiter=',')
    fig = plt.figure()
    plt.plot(fit_pars.T)
    plt.savefig(fout+'.png')
    plt.close(fig)
    res = res.append(dicm, ignore_index=True)
    dataset.save_sample(sample, dict(T1=fout+'.nii'))

fout = dirpath[0] +'res_simu.csv'
res.to_csv(fout)

dd = res[[ 'L1', 'MSE', 'ssim', 'corr', 'mean_DispP', 'rmse_Disp', 'rmse_DispTF']]
import seaborn as sns
sns.pairplot(dd)

#mot_separate
y_Disp, y_swalF, y_swalM, y_sudF, y_sudM = [], [], [], [], []
plt.figure()
for rpf in rp_files:
    fit_pars = pd.read_csv(rpf, header=None).values
    st=rpf
    temp = [pos for pos, char in enumerate(st) if char == "_"]
    y_Disp=int(st[temp[-13]+1:temp[-12]])/100
    y_Noise=int(st[temp[-11]+1:temp[-10]])/100
    y_swalF=np.floor(int(st[temp[-9]+1:temp[-8]])/100)
    y_swalM=int(st[temp[-7]+1:temp[-6]])/100
    y_sudF=np.floor(int(st[temp[-5]+1:temp[-4]])/100)
    y_sudM=int(st[temp[-3]+1:temp[-2]])/100

    dico_params = {
        "maxDisp": (y_Disp,y_Disp),"maxRot": (y_Disp,y_Disp),"noiseBasePars": (y_Noise,y_Noise),
        "swallowFrequency": (y_swalF,y_swalF+1), "swallowMagnitude": (y_swalM,y_swalM),
        "suddenFrequency": (y_sudF, y_sudF+1),"suddenMagnitude": (y_sudM, y_sudM),
        "verbose": True,
    }

    t = RandomMotionFromTimeCourse(**dico_params)
    t._calc_dimensions((100,20,50))
    fitP = t._simulate_random_trajectory()
    fitP = t.fitpars
    if True:# y_Disp>0:
        plt.figure()
        plt.plot(fit_pars.T)
        plt.plot(fitP.T,'--')



#test transforms
from torchio.transforms import RandomSpike
t = RandomSpike(num_spikes_range=(5,10), intensity_range=(0.1,0.2))
dataset = ImagesDataset(suj, transform=t)

for i in range(1,10):
    sample = dataset[0]
    fout='/tmp/toto{}_nb{}_I{}.nii'.format(i,sample['T1']['random_spike_num_spikes'],np.floor(sample['T1']['random_spike_intensity']*100))
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
    "keep_original": True,
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
