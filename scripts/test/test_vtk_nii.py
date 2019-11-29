import vtk
import numpy as np
import nibabel as nib

image_path = '/home/fernando/episurg/subjects/1412/mri/t1_post/assessors/1412_t1_post_on_mni.nii.gz'
image_path = '/tmp/tmp.nii.gz'

nii = nib.load(image_path)

image_reader = vtk.vtkNIFTIImageReader()
image_reader.SetFileName(image_path)
image_reader.Update()
image = image_reader.GetOutput()
header = image_reader.GetNIFTIHeader()

np.set_printoptions(precision=3, suppress=True)
print(nii.affine)
print(image_reader.GetQFormMatrix())
