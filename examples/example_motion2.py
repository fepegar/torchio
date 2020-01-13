from pprint import pprint
from torchio import Image, ImagesDataset, transforms, INTENSITY, LABEL
from torchvision.transforms import Compose

from torchio.transforms.augmentation.intensity.random_motion2 import   MotionSimTransform
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

for i  in range(0,1):

    sample = deepcopy(sample_orig)

    transformed = transforms(sample)
    name = 'mot'
    path = out_dir + f'{i}_{name}.nii.gz'
    dataset.save_sample(transformed, dict(T1=path))
