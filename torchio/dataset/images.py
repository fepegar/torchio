from pathlib import Path
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
        Each element in subjects_list is a dictionary with one mandatory key
        'image' and one optional key 'label'. The value for 'label' is the path
        to the label image. The value for 'image' is itself another dictionary
        whose keys are free to choose and whose values are paths to images.
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
        img = nib.load(str(path))

        # See https://github.com/nipy/dmriprep/issues/55#issuecomment-448322366
        data = np.array(img.dataobj).astype(np.float32)

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
        return data, img.affine

    @staticmethod
    def parse_subjects_list(subjects_list):
        def parse_path(path):
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(f'{path} not found')
        if not subjects_list:
            raise ValueError('Subjects list is empty')
        for subject_dict in subjects_list:
            for image_dict in subject_dict.values():
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
