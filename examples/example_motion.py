"""
Another way of getting this result is by running the command-line tool:

$ torchio-transform ~/Dropbox/MRI/t1.nii.gz RandomMotion /tmp/t1_motion.nii.gz --verbose --seed 42 --kwargs "degrees=10 translation=10 num_transforms=3 proportion_to_augment=1"

"""

from pprint import pprint
from torchio import Image, ImagesDataset, transforms, INTENSITY, LABEL, Subject


#subject = Subject(
#    Image('label', '~/Dropbox/MRI/t1_brain_seg.nii.gz', LABEL),
#    Image('t1', '~/Dropbox/MRI/t1.nii.gz', INTENSITY),
#)
subject = Subject(
    Image('t1', '/data/romain/HCPdata/suj_100307/T1w_1mm.nii.gz', INTENSITY),
    Image('label', '/data/romain/HCPdata/suj_100307/T1w_1mm.nii.gz', LABEL),
)

subjects_list = [subject]

dataset = ImagesDataset(subjects_list)

sample = dataset[0]
transform = transforms.RandomMotion(
    seed=2,
    degrees=0,
    translation=100,
    num_transforms=1,
    verbose=True,
    proportion_to_augment=1,
)
transformed = transform(sample)

pprint(transformed['t1']['random_motion_times'])
pprint(transformed['t1']['random_motion_degrees'])
pprint(transformed['t1']['random_motion_translation'])

dataset.save_sample(transformed, dict(t1='/tmp/t1_motion.nii.gz'))
dataset.save_sample(transformed, dict(label='/tmp/t1_brain_seg_motion.nii.gz'))
