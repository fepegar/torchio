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
import pandas as pd

from torchio.transforms import Interpolation
suj = [[ Image('T1', '/data/romain/HCPdata/suj_274542/mT1w_1mm.nii', INTENSITY), ]]
suj = [[ Image('T1', '/data/romain/data_exemple/suj_274542/mask_brain.nii', INTENSITY), ]]



def corrupt_data( x0, sigma= 5, amplitude=20, method='gauss'):
    fp = np.zeros((6, 200))
    x = np.arange(0,200)
    if method=='gauss':
        y = np.exp(-(x - x0) ** 2 / float(2 * sigma ** 2))*amplitude
    elif method == 'step':
        if x0<100:
            y = np.hstack((np.zeros((1,(x0-sigma))),
                            np.linspace(0,20,2*sigma+1).reshape(1,-1),
                            np.ones((1,((200-x0)-sigma-1)))*20 ))
        else:
            y = np.hstack((np.zeros((1,(x0-sigma))),
                            np.linspace(0,-20,2*sigma+1).reshape(1,-1),
                            np.ones((1,((200-x0)-sigma-1)))*-20 ))
    fp[1,:] = y
    return fp

dico_params = {    "fitpars": None,  "verbose": True, "displacement_shift":False }

x0=np.hstack((np.arange(90,102,2),np.arange(101,105,1)))
x0=[100]

dirpath = ['/data/romain/data_exemple/motion_gauss']
#plt.ioff()
#for s in [1, 5, 10, 20 ]:
for s in [2,4,6] : #[1, 3 , 5 , 8, 10 , 12, 15, 20 , 25 ]:
    for xx in x0:
        fp = corrupt_data(xx, sigma=s,method='step')
        dico_params['fitpars'] = fp
        t = RandomMotionFromTimeCourse(**dico_params)
        dataset = ImagesDataset(suj, transform=t)
        sample = dataset[0]
        fout = dirpath[0] + '/mask_mot_no_shift_s{:02d}_x{}'.format(s,xx)
        fit_pars = t.fitpars
        fig = plt.figure()
        plt.plot(fit_pars.T)
        plt.savefig(fout+'.png')
        plt.close(fig)
        dataset.save_sample(sample, dict(T1=fout+'.nii'))


dataset = ImagesDataset(suj)
so=dataset[0]
image = so['T1']['data'][0]
tfi = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)
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

for s in [1,2, 3, 4, 5 , 8, 10 , 12, 15, 20 , 2500 ]:
    fp = corrupt_data(50, sigma=s, method='gauss')
    dico_params['fitpars'] = fp
    t = RandomMotionFromTimeCourse(**dico_params)

    t._calc_dimensions(sample['T1']['data'][0].shape)
    fitpars_interp = t._interpolate_space_timing(t.fitpars)
    trans = fitpars_interp[1,0,:]
    #plt.figure(); plt.plot(trans.reshape(-1))
    print(np.sum(trans*sum_intensity_abs)/np.sum(sum_intensity_abs))

np.sum()






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
               "verbose": True, "keep_original": True, "compare_to_original": True}
t = RandomMotionFromTimeCourse(**dico_params)
dataset = ImagesDataset(suj, transform=t)

res = pd.DataFrame()
dirpath = ['/data/romain/HCPdata/suj_274542/check2/']
plt.ioff()
for i in range(100):
    sample = dataset[0]
    dicm = sample['T1']['metrics']
    dics = sample['T1']['simu_param']
    fout = dirpath[0]  +'mot_sim{}'.format(np.floor(dicm['ssim']*10000))
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
