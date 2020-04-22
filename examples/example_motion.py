"""
Another way of getting this result is by running the command-line tool:

$ torchio-transform ~/Dropbox/MRI/t1.nii.gz RandomMotion /tmp/t1_motion.nii.gz --seed 42 --kwargs "degrees=10 translation=10 num_transforms=3"

"""

from pprint import pprint
from torchio import Image, ImagesDataset, transforms, INTENSITY, LABEL, Subject

subject = Subject(
    label=Image('~/Dropbox/MRI/t1_brain_seg.nii.gz', LABEL),
    t1=Image('~/Dropbox/MRI/t1.nii.gz', INTENSITY),
)
subjects_list = [subject]

dataset = ImagesDataset(subjects_list)
sample = dataset[0]
transform = transforms.RandomMotion(
    seed=42,
    degrees=10,
    translation=10,
    num_transforms=3,
)
transformed = transform(sample)

_, random_parameters = transformed.history[0]

pprint(random_parameters['t1']['times'])
pprint(random_parameters['t1']['degrees'])
pprint(random_parameters['t1']['translation'])

dataset.save_sample(transformed, dict(t1='/tmp/t1_motion.nii.gz'))
dataset.save_sample(transformed, dict(label='/tmp/t1_brain_seg_motion.nii.gz'))
