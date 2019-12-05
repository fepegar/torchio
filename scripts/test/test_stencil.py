from tempfile import NamedTemporaryFile
from pathlib import Path
import numpy as np
import nibabel as nib

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


if __name__ == "__main__":
    main()
