from pathlib import Path
from collections.abc import Sequence

import nrrd
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from ..utils import get_stem


class ImagesDataset(Dataset):
    def __init__(
            self,
            subjects_list,
            transform=None,
            verbose=False,
            ):
        """
        Each element of subjects_list is a dictionary:
        subject = {
            'one_image': dict(
                path=path_to_one_image,
                type=torchio.INTENSITY,
            ),
            'another_image': dict(
                path=path_to_another_image,
                type=torchio.INTENSITY,
            ),
            'a_label': dict(
                path=path_to_a_label,
                type=torchio.LABEL,
            ),
        }
        See examples/example_multimodal.py for -obviously- an example.
        """
        self.parse_subjects_list(subjects_list)
        self.subjects_list = subjects_list
        self.transform = transform
        self.verbose = verbose

    def __len__(self):
        return len(self.subjects_list)

    def __getitem__(self, index):
        subject_dict = self.subjects_list[index]
        sample = {}
        for image_name, image_dict in subject_dict.items():
            image_path = image_dict['path']
            data, affine = self.load_image(image_path)
            image_sample_dict = dict(
                data=data,
                path=str(image_path),
                affine=affine,
                stem=get_stem(image_path),
                type=image_dict['type'],
            )
            sample[image_name] = image_sample_dict

        # Apply transform (this is usually the major bottleneck)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def load_image(self, path, add_channels_dim=True):
        if self.verbose:
            print(f'Loading {path}...')
        path = Path(path).expanduser()

        if '.nii' in path.suffixes:
            nii = nib.load(str(path))
            # See https://github.com/nipy/dmriprep/issues/55#issuecomment-448322366
            data = np.array(nii.dataobj).astype(np.float32)
            affine = nii.affine
        elif '.nrrd' in path.suffixes:
            data, header = nrrd.read(path)
            data = data.astype(np.float32)
            affine = np.eye(4)
            affine[:3, :3] = header['space directions'].T
            affine[:3, 3] = header['space origin']
            lps_to_ras = np.diag((-1, -1, 1, 1))
            affine = np.dot(lps_to_ras, affine)

        if self.verbose:
            print(f'Loaded array with shape {data.shape}')
        num_dimensions = data.ndim
        if num_dimensions > 3:
            message = (
                f'Processing of {num_dimensions}D volumes not supported.'
                f' {path} has shape {data.shape}'
            )
            raise NotImplementedError(message)
        data = data[np.newaxis, ...] if add_channels_dim else data
        return data, affine

    @staticmethod
    def parse_subjects_list(subjects_list):
        def parse_path(path):
            path = Path(path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f'{path} not found')
        if not isinstance(subjects_list, Sequence):
            raise ValueError(
                f'Subject list must be a sequence, not {type(subjects_list)}')
        if not subjects_list:
            raise ValueError('Subjects list is empty')
        for element in subjects_list:
            if not isinstance(element, dict):
                raise ValueError(
                    f'All elements must be dictionaries, not {type(element)}')
            subject_dict = element
            for image_dict in subject_dict.values():
                for key in ('path', 'type'):
                    if key not in image_dict:
                        raise ValueError(
                            f'"{key}" not found in image dict {image_dict}')
                parse_path(image_dict['path'])

    @staticmethod
    def save_sample(sample, output_paths_dict):
        # TODO: adapt to new subjects_list structure
        for key, output_path in output_paths_dict.items():
            data = sample[key].squeeze()
            affine = sample['affine']
            nii = nib.Nifti1Image(data, affine)
            nii.header['qform_code'] = 1
            nii.header['sform_code'] = 0
            nii.to_filename(str(output_path))
