import vtk
import numpy as np
import nibabel as nib

image_path = '/home/fernando/episurg/subjects/1412/mri/t1_post/assessors/1412_t1_post_on_mni.nii.gz'
image_path = '/tmp/tmp.nii.gz'
image_path = '/tmp/transform/1395_gray_matter_left_label.nii.gz'

nii = nib.load(image_path)

image_reader = vtk.vtkNIFTIImageReader()
image_reader.SetFileName(image_path)
image_reader.Update()
image = image_reader.GetOutput()
header = image_reader.GetNIFTIHeader()

np.set_printoptions(precision=3, suppress=True)

qform_nib = nii.get_qform()
qform_vtk = image_reader.GetQFormMatrix()

print(qform_nib)
print(qform_vtk)

transform_nib = None
transform_vtk = vtk.vtkTransform()
transform_vtk.SetMatrix(qform_vtk)

v = 2, 3, 4

x_nib = nib.affines.apply_affine(nii.affine, v)
x_vtk = transform_vtk.TransformPoint(v)

print(x_nib)
print(x_vtk)
