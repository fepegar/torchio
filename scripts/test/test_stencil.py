import numpy as np
import nibabel as nib

import vtk
from vtk.numpy_interface import dataset_adapter as dsa

tmp_path = '/tmp/tmp.nii.gz'
nii = nib.load('/tmp/transform/1395_gray_matter_left_label.nii.gz')
data = nii.get_data()
data[:] = 1
nii = nib.Nifti1Image(data, nii.affine)
nii.header['sform_code'] = 0
nii.header['qform_code'] = 1
nii.to_filename(tmp_path)
image_path = tmp_path

# image_path = '/tmp/transform/cube.nii.gz'
# mesh_path = '/tmp/transform/sphere_100.vtp'
mesh_path = '/tmp/test.vtp'

image_reader = vtk.vtkNIFTIImageReader()
image_reader.SetFileName(image_path)
image_reader.Update()
image = image_reader.GetOutput()
header = image_reader.GetNIFTIHeader()

pd_reader = vtk.vtkXMLPolyDataReader()
pd_reader.SetFileName(mesh_path)
pd_reader.Update()
pd_ras = pd_reader.GetOutput()

transform_poly_data = vtk.vtkTransformPolyDataFilter()
transform = vtk.vtkTransform()

ras_to_ijk = vtk.vtkMatrix4x4()
qform_inv = np.linalg.inv(nii.affine)
for i in range(3):
    for j in range(4):
        ras_to_ijk.SetElement(i, j, qform_inv[i, j])
transform.SetMatrix(ras_to_ijk)
transform_poly_data.SetTransform(transform)
transform_poly_data.SetInputData(pd_ras)
transform_poly_data.Update()
pd_ijk = transform_poly_data.GetOutput()


# ras_to_lps = vtk.vtkMatrix4x4()
# ras_to_lps.SetElement(0, 0, -1)
# ras_to_lps.SetElement(1, 1, -1)
# transform.SetMatrix(ras_to_lps)
# transform_poly_data.SetTransform(transform)
# transform_poly_data.SetInputData(pd_ras)
# transform_poly_data.Update()
# pd_lps = transform_poly_data.GetOutput()

# ijk_to_lps = image_reader.GetQFormMatrix()
# lps_to_ijk = vtk.vtkMatrix4x4()
# lps_to_ijk.DeepCopy(ijk_to_lps)
# lps_to_ijk.Invert()
# transform.SetMatrix(lps_to_ijk)
# transform_poly_data.SetTransform(transform)
# transform_poly_data.SetInputData(pd_lps)
# transform_poly_data.Update()
# pd_ijk = transform_poly_data.GetOutput()

# normalFilter = vtk.vtkPolyDataNormals()
# normalFilter.SetInputData(pd_ijk)
# normalFilter.ConsistencyOn()
# normalFilter.Update()
# pd_ijk = normalFilter.GetOutput()

polyDataToImageStencil = vtk.vtkPolyDataToImageStencil()
polyDataToImageStencil.SetInputData(pd_ijk)
polyDataToImageStencil.SetOutputSpacing(image.GetSpacing())
polyDataToImageStencil.SetOutputOrigin(image.GetOrigin())
polyDataToImageStencil.SetOutputWholeExtent(image.GetExtent())
polyDataToImageStencil.Update()

stencil = vtk.vtkImageStencil()
stencil.SetInputData(image)
stencil.SetStencilData(polyDataToImageStencil.GetOutput())
stencil.SetBackgroundValue(0)
stencil.Update()
image_output = stencil.GetOutput()

result_path = '/tmp/transform/sphere_100.nii.gz'

ds = dsa.WrapDataObject(image_output)
arr = ds.PointData['NIFTI'].reshape(data.shape, order='F')  # C didn't work
print(arr.size)
print(np.count_nonzero(arr))
output_nii = nib.Nifti1Image(arr, nii.affine)
output_nii.header['sform_code'] = 0
output_nii.header['qform_code'] = 1
output_nii.to_filename(result_path)

# image_writer = vtk.vtkNIFTIImageWriter()
# image_writer.SetNIFTIHeader(header)
# image_writer.SetFileName(result_path)
# image_writer.SetInputData(image_output)
# image_writer.Write()


# image_reader = vtk.vtkNIFTIImageReader()
# image_reader.SetFileName(image_path)
# image_reader.Update()
# image = image_reader.GetOutput()
# header = image_reader.GetNIFTIHeader()
# print(header)

# image_reader = vtk.vtkNIFTIImageReader()
# image_reader.SetFileName(result_path)
# image_reader.Update()
# image = image_reader.GetOutput()
# header = image_reader.GetNIFTIHeader()
# print(header)
