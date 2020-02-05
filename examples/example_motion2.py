from pprint import pprint
from torchio import Image, ImagesDataset, transforms, INTENSITY, LABEL
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
import numpy as np

from torchio.transforms.augmentation.intensity.random_motion2 import MotionSimTransform
from copy import deepcopy
from nibabel.viewers import OrthoSlicer3D as ov

np.random.seed(12)

out_dir = '/data/romain/data_exemple/augment/'
suj = [[
    Image('T1','/data/romain/HCPdata/suj_100307/T1w_1mm.nii.gz',INTENSITY),
    Image('mask','/data/romain/HCPdata/suj_100307/brain_mT1w_1mm.nii',LABEL)
     ]]
t = MotionSimTransform(std_rotation_angle=3, std_translation=2, nufft=True, proc_scale=0, verbose=True, freq_encoding_dim=(0,),
                       mvt_param=[0, 0, 0, 1, 10, 0])

transforms = Compose([t])

dataset = ImagesDataset(suj,transform=transforms)


sample = dataset[0]
dataset.save_sample(sample, dict(T1='/tmp/toto_no_center.nii'))



sample_orig=dataset[0]

for i  in range(0,1):

    sample = deepcopy(sample_orig)

    transformed = transforms(sample)
    name = 'mot'
    path = out_dir + f'{i}_{name}.nii.gz'
    dataset.save_sample(transformed, dict(T1=path))


########################################################### T E S T S with txt
import pandas as pd
import nibabel as nb
import numpy as np

# Read mvt params (6 x 219)
rpdir = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/simulation/test_motion'
rpfile = [ #rpdir+'/three_swallow/rp_Motion_RMS_63_Disp_0_Noise_0_swalF_300_swalM_400_sudF_0_sudM_0_PF_0_mot.txt',
          rpdir+'/perlin/rp_Motion_RMS_209_Disp_400_Noise_300_swalF_0_swalM_0_sudF_0_sudM_0_PF_0_motDim2.txt',
          rpdir+'/onesudden/rp_Motion_RMS_219_Disp_0_Noise_0_swalF_0_swalM_0_sudF_100_sudM_400_PF_0_mot.txt']
outdir = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/simulation/test_motion/out_python/'
outfile = [#outdir + 'three_swallowDim1.nii.gz',
            outdir + 'perfin.nii.gz', outdir + 'onesudden.nii.gz']

# Read MRI data
mri_data = nb.load("/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/simulation/test_motion/T1w_1mm.nii.gz")
image = mri_data.get_data()
# Squeezed img
squeezed_img = np.squeeze(image)

for fin, fout in zip(rpfile,outfile):

    data = pd.read_csv(fin, header=None)
    data = data.drop(columns=[218])
    #Extract separately parameters of translation and rotation
    trans_data, rot_data = data[:3].values.reshape((3, 1, 218, -1)), data[3:].values.reshape((3, 1, 218, -1))

    #Broadcast translation and rotation parameters to voxel level params
    ones_multiplicator = np.ones([3] + list(squeezed_img.shape))
    translations, rotations = trans_data * ones_multiplicator, rot_data * ones_multiplicator
    ##To voxel-wise params
    translations, rotations = translations.reshape((3, -1)), rotations.reshape((3, -1))
    rotations = np.radians(rotations)

    #### TRANSFORMATION
    t = MotionSimTransform(std_rotation_angle=5, std_translation=10, nufft=True)
    t.frequency_encoding_dim = 1 #On fixe la dimension d'encodage de fréquence
    t._calc_dimensions(squeezed_img.shape)
    # Set rotations and translation params
    t.translations, t.rotations = translations, rotations
    im_freq_domain = t._fft_im(squeezed_img)

    if True:
        #Translation
        translated_freq = t._translate_freq_domain(im_freq_domain)
        #translated_img = t._ifft_im(translated_freq)
        #corr_trans = abs(translated_img)
        #Rotation
        rotated_freq = t._nufft(translated_freq)
        corr_rot = abs(rotated_freq) / rotated_freq.size
        # Save data
        corr_rot_img = nb.Nifti1Image(corr_rot, mri_data.affine)

    else:
        rotated_freq = t._nufft(im_freq_domain)
        corr_rot = abs(rotated_freq) / rotated_freq.size
        corr_rot_freq_domain = t._fft_im(corr_rot)

        #Translation
        translated_freq = t._translate_freq_domain(corr_rot_freq_domain)
        translated_img = t._ifft_im(translated_freq)
        corr_trans = abs(translated_img)

        #Save data
        corr_rot_img = nb.Nifti1Image(corr_trans, mri_data.affine)

    corr_rot_img.to_filename(fout)
