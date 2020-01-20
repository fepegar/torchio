from torchio import ImagesDataset, Image, INTENSITY, LABEL
from torchio.data.io import write_image
from torchio.data import  get_subject_list_and_csv_info_from_data_prameters
from torchio.transforms import RandomFlip, RandomAffine, RandomElasticDeformation, \
    HistogramStandardization, HistogramEqualize, HistogramRandomChange,\
    Interpolation, RandomMotion, RandomBiasField, Rescale
from torchvision.transforms import Compose
import torch

import sys
import numpy as np
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D as ov
sys.path.extend(['/data/romain/toolbox_python/romain/cnnQC_pytorch'])
from utils_file import gdir, gfile, get_parent_path

import importlib
import torchio.transforms.preprocessing.histogram_standardization
importlib.reload(torchio.transforms.preprocessing.histogram_standardization)
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

test = {'T1': {'csv_file':'/data/romain/HCPdata/Motion_brain_ms_train_hcp400.csv'} }
conditions = [("corr", "<", 0.98), ("|", "noise", "==", 1)]
subjects_dict, info = get_subject_list_and_csv_info_from_data_prameters(test, fpath_idx='filename', conditions=conditions,shuffle_order=True)

data_parameters = {'image': {'csv_file': '/data/romain/data_exemple/file_ms.csv', 'type': torchio.INTENSITY},
                   'label1': {'csv_file': '/data/romain/data_exemple/file_p1.csv', 'type': torchio.LABEL},
                   'label2': {'csv_file': '/data/romain/data_exemple/file_p2.csv', 'type': torchio.LABEL},
                   'label3': {'csv_file': '/data/romain/data_exemple/file_p3.csv', 'type': torchio.LABEL},
                   'sampler': {'csv_file': '/data/romain/data_exemple/file_mask.csv', 'type': torchio.SAMPLING_MAP}}
paths_dict, info = get_subject_list_and_csv_info_from_data_prameters(data_parameters) #,shuffle_order=False)

landmarks_file = '/data/romain/data_exemple/landmarks_hcp100.npy'
transforms = (HistogramStandardization(landmarks_file, mask_field_name='sampler'),)

transforms = (RandomElasticDeformation(num_control_points=8, proportion_to_augment=1, deformation_std=25, image_interpolation=Interpolation.BSPLINE),)
transforms = (RandomMotion(seed=42, degrees=0, translation=15, num_transforms=2, verbose=True,proportion_to_augment=1),)
transforms = (RandomBiasField(coefficients_range=(-0.5, 0.5),order=3), )

transform = Compose(transforms) #should be done in ImagesDataset
dataset = ImagesDataset(paths_dict, transform=transform)
dataset_not = ImagesDataset(paths_dict, transform=None)
dataload = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=1)
dataloadnot = torch.utils.data.DataLoader(dataset_not, num_workers=0, batch_size=1)

ddd = dataset[0] #next(iter(dataset))
ii = np.squeeze( ddd['image']['data'][0], axis=1)

ddno = dataset_not[0] #

dd= next(iter(dataload))
ddno = next(iter(dataloadnot))

ii = np.squeeze( dd['image']['data'][0,0,:],axis=1)
iio = np.squeeze( ddno['image']['data'][0,0,:],axis=1)

ov(ii)
ov(iio)

#save
sample = dataset[0]
output = dict(image=Path('/tmp/test_im.nii.gz'),label1=Path('/tmp/test_p1.nii.gz'),label2=Path('/tmp/test_p2.nii.gz'))
dataset.save_sample(sample, output)


#explore
out_dir = '/data/romain/data_exemple/augment/'
suj = [[
    Image('T1','/data/romain/data_exemple/nifti_proc/PRISMA_MBB_DB/2017_03_07_DEV_236_MBB_DB_Pilote02/anat_S02_t1mpr_SAG_NSel_S176/cat12/s_S02_t1mpr_SAG_NSel_S176.nii.gz',INTENSITY),
    Image('mask','/data/romain/data_exemple/nifti_proc/PRISMA_MBB_DB/2017_03_07_DEV_236_MBB_DB_Pilote02/anat_S02_t1mpr_SAG_NSel_S176/cat12/mask_brain_erode_dilate.nii.gz',LABEL)
     ]]
transforms = Compose((RandomBiasField(coefficients_range=(-0.5, 0.5),order=3, verbose=True), ))
landmarks_file = '/data/romain/data_exemple/landmarks_hcp300_res100.txt'
transforms = Compose((HistogramStandardization(landmarks_file, verbose=True, masking_method='mask'), Rescale(masking_method='mask',verbose=True)))
transforms = HistogramEqualize(verbose=True,masking_method='mask')
transforms = Rescale(verbose=True,masking_method='mask')

torch.manual_seed(12)
np.random.seed(12)
transforms = Compose((HistogramRandomChange(verbose=True, masking_method='mask'), Rescale(masking_method='mask',verbose=True)))
dataset1 = ImagesDataset(suj, transform=transforms )

for i in range(1,10):
    dd = dataset1[0]
    name = 'histo'
    out_path = out_dir + f'{i}_{name}.nii.gz'
    dataset1.save_sample(dd, dict(T1=out_path))

ii = dd['T1']['data'][0]

ii[ii<0.1]=0
iif = ii.flatten()
iif = iif[iif>0]
plt.hist(iif,bins=128)


#fernando random motion
dataset = ImagesDataset(suj)
transforms = [
    #ZNormalization(verbose=verbose),
    RandomMotion(proportion_to_augment=1, seed=1, verbose=True),
]
sample = dataset[0]
for i, transform in enumerate(transforms):
    transformed = transform(sample)
    name = transform.__class__.__name__
    path = f'/tmp/{i}_{name}_abs.nii.gz'
    dataset.save_sample(transformed, dict(T1=path))

#histo normalization
from torchio.transforms.preprocessing.histogram_standardization import train, normalize

suj = gdir('/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/HCPdata','^suj')
allfiles = gfile(suj,'^T1w_1mm.nii.gz')
allfiles_mask = gfile(suj,'^brain_T1w_1mm.nii.gz')
testf = allfiles[0:300]
outname ='/data/romain/data_exemple/landmarks_hcp300_res100.npy'
#outname ='/data/romain/data_exemple/landmarks_hcp300_res100_cutof01.npy'

landmark = train(testf, output_path=outname, mask_path=allfiles_mask, cutoff=(0, 1))

nii = nib.load(testf[0]).get_fdata(dtype=np.float32)
niim = normalize(nii, landmark)

perc_database=np.load(outname)


mm = np.mean(perc_database, axis=1)
mr = np.tile(mm,(perc_database.shape[1], 1)).T
perc_m = perc_database / mr   # less dependant on the extrem (compare to min max norm)

perc2 = (perc_database - np.tile(np.min(perc_database, axis=1), (perc_database.shape[1], 1)).T) / \
        (np.tile(np.max(perc_database, axis=1), (perc_database.shape[1], 1)).T -
         np.tile(np.min(perc_database, axis=1), (perc_database.shape[1], 1)).T)*100
perc_mean = np.mean(perc2, axis=0)  # this is equivalent to what is done in __averaged_mapping

resc=pd.read_csv('/home/romain.valabregue/datal/HCPdata/stats/res_cat12_QC_300_suj.csv')
voltot = resc['vol_W']+resc['vol_C']+resc['vol_G']


#randome histo line
out_dir = '/data/romain/data_exemple/augment/'
suj = [[
    Image('T1','/data/romain/HCPdata/suj_100307/T1w_1mm.nii.gz',INTENSITY),
    Image('mask','/data/romain/HCPdata/suj_100307/brain_mT1w_1mm.nii',LABEL)
     ]]
suj = [[
    Image('T1','/data/romain/data_exemple/nifti_proc/PRISMA_MBB_DB/2017_03_07_DEV_236_MBB_DB_Pilote02/anat_S02_t1mpr_SAG_NSel_S176/cat12/s_S02_t1mpr_SAG_NSel_S176.nii.gz',INTENSITY),
    Image('mask','/data/romain/data_exemple/nifti_proc/PRISMA_MBB_DB/2017_03_07_DEV_236_MBB_DB_Pilote02/anat_S02_t1mpr_SAG_NSel_S176/cat12/mask_brain_erode_dilate.nii.gz',LABEL)
     ]]
dataset = ImagesDataset(suj)

from copy import deepcopy
transfo = Compose(( Rescale(masking_method='mask', verbose=True),))
transfo =  Rescale( out_min_max=(0,1), verbose=True)

sample_orig = dataset[0]
sample = deepcopy(sample_orig)

sample = transfo(sample)
path = out_dir + 'orig.nii.gz'
dataset.save_sample(sample, dict(T1=path))

nb_point_ini = 50
nb_smooth = 5
#yall = perc2
for i  in range(0,10):

    y2 = get_random_curve(nb_point_ini,nb_smooth)
    #y2 = get_curve_for_sample(yall)

    transforms = Compose(
        (HistogramStandardization(dict(T1=y2), verbose=True, masking_method='mask'), Rescale( verbose=True, masking_method='mask')))
        #(HistogramStandardization(y2, verbose=True, mask_field_name='mask'), Rescale(masking_method='mask', verbose=True)))

    sample = deepcopy(sample_orig)

    transformed = transforms(sample)
    name = 'histo'
    path = out_dir + f'{i}_{name}.nii.gz'
    dataset.save_sample(transformed, dict(T1=path))









#test motion ghiles
from torchio.transforms.augmentation.intensity.random_motion2 import  import  MotionSimTransform
from copy import deepcopy
out_dir = '/data/romain/data_exemple/augment/'
suj = [[
    Image('T1','/data/romain/HCPdata/suj_100307/T1w_1mm.nii.gz',INTENSITY),
    Image('mask','/data/romain/HCPdata/suj_100307/brain_mT1w_1mm.nii',LABEL)
     ]]
t = MotionSimTransform(std_rotation_angle=3, std_translation=2, nufft=True, proc_scale=0, verbose=True, freq_encoding_dim=(2,))
transforms = Compose([t])

dataset = ImagesDataset(suj)
sample_orig=dataset[0]

for i  in range(0,4):

    sample = deepcopy(sample_orig)

    transformed = transforms(sample)
    name = 'mot'
    path = out_dir + f'{i}_{name}.nii.gz'
    dataset.save_sample(transformed, dict(T1=path))



sample = dataset[0]
image_data = np.squeeze(sample['T1']['data'])[..., np.newaxis, np.newaxis]
original_image = np.squeeze(image_data[:, :, :, 0, 0])

mot = MotionSimTransform(proc_scale=0)
mot._calc_dimensions(original_image.shape)

tran, rot = mot._simul_motion(200)














# old on from fernando test_stencil

from tempfile import NamedTemporaryFile
from pathlib import Path

import vtk
from vtk.numpy_interface import dataset_adapter as dsa


def poly_data_to_nii(poly_data, reference_path, result_path):
    """
    TODO: stop reading and writing so much stuff
    Write to buffer? Bytes? Investigate this
    """
    nii = nib.load(str(reference_path))
    image_stencil_array = np.ones(nii.shape, dtype=np.uint8)
    image_stencil_nii = nib.Nifti1Image(image_stencil_array, nii.get_qform())

    with NamedTemporaryFile(suffix='.nii') as f:
        stencil_path = f.name
        image_stencil_nii.to_filename(stencil_path)
        image_stencil_reader = vtk.vtkNIFTIImageReader()
        image_stencil_reader.SetFileName(stencil_path)
        image_stencil_reader.Update()

    image_stencil = image_stencil_reader.GetOutput()
    xyz_to_ijk = image_stencil_reader.GetQFormMatrix()
    if xyz_to_ijk is None:
        import warnings
        warnings.warn('No qform found. Using sform')
        xyz_to_ijk = image_stencil_reader.GetSFormMatrix()
    xyz_to_ijk.Invert()

    transform = vtk.vtkTransform()
    transform.SetMatrix(xyz_to_ijk)

    transform_poly_data = vtk.vtkTransformPolyDataFilter()
    transform_poly_data.SetTransform(transform)
    transform_poly_data.SetInputData(poly_data)
    transform_poly_data.Update()
    pd_ijk = transform_poly_data.GetOutput()

    polyDataToImageStencil = vtk.vtkPolyDataToImageStencil()
    polyDataToImageStencil.SetInputData(pd_ijk)
    polyDataToImageStencil.SetOutputSpacing(image_stencil.GetSpacing())
    polyDataToImageStencil.SetOutputOrigin(image_stencil.GetOrigin())
    polyDataToImageStencil.SetOutputWholeExtent(image_stencil.GetExtent())
    polyDataToImageStencil.Update()

    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(image_stencil)
    stencil.SetStencilData(polyDataToImageStencil.GetOutput())
    stencil.SetBackgroundValue(0)
    stencil.Update()

    image_output = stencil.GetOutput()

    data_object = dsa.WrapDataObject(image_output)
    array = data_object.PointData['NIFTI']
    array = array.reshape(nii.shape, order='F')  # C didn't work :)
    array = check_qfac(nii, array)

    output_nii = nib.Nifti1Image(array, nii.affine)
    output_nii.header['sform_code'] = 0
    output_nii.header['qform_code'] = 1
    output_nii.to_filename(result_path)


def check_qfac(nifti, array):
    """
    See https://vtk.org/pipermail/vtk-developers/2016-November/034479.html
    """
    qfac = nifti.header['pixdim'][0]
    if qfac not in (-1, 1):
        raise ValueError(f'Unknown qfac value: {qfac}')
    elif qfac == -1:
        array = array[..., ::-1]
    return array


def main():
    mesh_path = '/tmp/test.vtp'
    pd_reader = vtk.vtkXMLPolyDataReader()
    pd_reader.SetFileName(mesh_path)
    pd_reader.Update()
    poly_data = pd_reader.GetOutput()

    reference_path = Path('~/Dropbox/MRI/t1_on_mni.nii.gz').expanduser()
    # reference_path = Path('/tmp/transform/1395_gray_matter_left_label.nii.gz')

    result_path = '/tmp/result_label.nii.gz'

    poly_data_to_nii(poly_data, reference_path, result_path)


