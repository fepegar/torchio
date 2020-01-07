from pathlib import Path
from collections.abc import Sequence
from torch.utils.data import Dataset
from ..utils import get_stem
from ..io import read_image, write_image


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
        #paths_dict = paths_dict.copy()
        #self.sujid = paths_dict.pop('sujid', None) #this will remove field sujid if exist
        #self.parse_paths_dict(paths_dict)
        #self.paths_dict = paths_dict

        self.parse_subjects_list(subjects_list)
        self.subjects_list = subjects_list
        self.transform = transform
        self.verbose = verbose

    def __len__(self):
        return len(self.subjects_list)

    def __getitem__(self, index):
        subject_dict = self.subjects_list[index]
        sample = {}

        # for key in self.paths_dict:
        #     data, affine, image_path = self.load_image(key, index)
        #     sample[key] = data
        #     if key == 'image':
        #         image_dict = dict(
        #             path=str(image_path),
        #             affine=affine,
        #             stem=get_stem(image_path),
        #             sujid=self.sujid[index] if self.sujid is not None else 'TODO',
        #         )
        #         sample.update(image_dict)
        #
        # label_name = [kk for kk in sample if ('label' in kk) ]
        # if len(label_name) > 1:
        #     list_label = [sample[kkk].squeeze() for kkk in label_name]
        #     for kkk in label_name :
        #         del sample[kkk]  #remove label_1 label_2 ... entery
        #     sample['label'] = np.stack(list_label)

        for image_name, image_dict in subject_dict.items():
            image_path = image_dict['path']
            tensor, affine = self.load_image(image_path)
            image_sample_dict = dict(
                data=tensor,
                path=str(image_path),
                affine=affine,
                stem=get_stem(image_path),
                type=image_dict['type'],
            )
            sample[image_name] = image_sample_dict

        # Apply transform (this is usually the bottleneck)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def load_image(self, path):
        if self.verbose:
            print(f'Loading {path}...')
        tensor, affine = read_image(path)
        if self.verbose:
            print(f'Loaded array with shape {tensor.shape}')
        return tensor, affine

    @staticmethod
    def parse_subjects_list(subjects_list):
        def parse_path(path):
            path = Path(path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f'{path} not found')
        if not isinstance(subjects_list, Sequence):
            raise TypeError(
                f'Subject list must be a sequence, not {type(subjects_list)}')
        if not subjects_list:
            raise ValueError('Subjects list is empty')
        for element in subjects_list:
            if not isinstance(element, dict):
                raise TypeError(
                    f'All elements must be dictionaries, not {type(element)}')
            if not element:
                raise ValueError(f'Element seems empty: {element}')
            subject_dict = element
            for image_name, image_dict in subject_dict.items():
                if not isinstance(image_dict, dict):
                    raise TypeError(
                        f'Type {type(image_dict)} found for {image_name},'
                        ' instead of type dict'
                    )
                for key in ('path', 'type'):
                    if key not in image_dict:
                        raise KeyError(
                            f'"{key}" key not found'
                            f' in image dict {image_dict}')
                parse_path(image_dict['path'])

    @staticmethod
    def save_sample(sample, output_paths_dict):
        for key, output_path in output_paths_dict.items():
            tensor = sample[key]['data'].squeeze()
            affine = sample[key]['affine']
            write_image(tensor, affine, output_path)
