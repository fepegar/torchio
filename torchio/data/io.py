from pathlib import Path
import nrrd
import torch
import numpy as np
import nibabel as nib


def parse_path(path):
    path = Path(path).expanduser()
    suffixes = path.suffixes
    if '.nii' not in suffixes and '.nrrd' not in suffixes:
        raise ValueError(
            'Image suffixes must contain ".nii" or ".nrrd",'
            f' but they are {suffixes}')
    return path


def read_image(path):
    path = parse_path(path)
    suffixes = path.suffixes
    if '.nii' in suffixes:
        read = _read_nifti
    elif '.nrrd' in suffixes:
        read = _read_nrrd
    else:
        raise NotImplementedError(
            f'Reading not implemented for this format: "{path}"')
    return read(path)


def _read_nifti(path):
    nii = nib.load(str(path), mmap=False)
    ndims = len(nii.shape)
    assert ndims == 3
    data = nii.get_fdata(dtype=np.float32)
    tensor = torch.from_numpy(data)
    affine = nii.affine
    return tensor, affine


def _read_nrrd(path):
    data, header = nrrd.read(path)
    data = data.astype(np.float32)
    tensor = torch.from_numpy(data)
    affine = np.eye(4)
    affine[:3, :3] = header['space directions'].T
    affine[:3, 3] = header['space origin']
    lps_to_ras = np.diag((-1, -1, 1, 1))
    affine = np.dot(lps_to_ras, affine)
    return tensor, affine


def write_image(tensor, affine, path):
    path = Path(path)
    suffixes = path.suffixes
    if '.nii' in suffixes:
        write = _write_nifti
    elif '.nrrd' in suffixes:
        write = _write_nrrd
    else:
        raise NotImplementedError(
            f'Writing not implemented for this format: "{path}"')
    write(tensor, affine, path)


def _write_nifti(tensor, affine, path):
    nii = nib.Nifti1Image(tensor.numpy(), affine)
    nii.header['qform_code'] = 1
    nii.header['sform_code'] = 0
    nii.to_filename(str(path))


def _write_nrrd(tensor, affine, path):
    raise NotImplementedError
