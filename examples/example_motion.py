"""
Another way of getting this result is by running the command-line tool:

$ torchio-transform ~/Dropbox/MRI/t1.nii.gz RandomMotion /tmp/t1_motion.nii.gz --seed 42 --kwargs "degrees=10 translation=10 num_transforms=3"

"""

from pprint import pprint
from torchio import Image, ImagesDataset, transforms, INTENSITY, LABEL, Subject


#subject = Subject(
#    Image('label', '~/Dropbox/MRI/t1_brain_seg.nii.gz', LABEL),
#    Image('t1', '~/Dropbox/MRI/t1.nii.gz', INTENSITY),
#)
dic_suj = {'t1': Image('/data/romain/HCPdata/suj_100307/T1w_1mm.nii.gz', INTENSITY),
           'label': Image('/data/romain/HCPdata/suj_100307/T1w_1mm.nii.gz', LABEL)}
subject = Subject(dic_suj)

subject = Subject(
    t1 = Image('/data/romain/HCPdata/suj_100307/T1w_1mm.nii.gz', INTENSITY),
    label = Image('/data/romain/HCPdata/suj_100307/T1w_1mm.nii.gz', LABEL),
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

_, random_parameters = transformed.history[0]

pprint(random_parameters['t1']['times'])
pprint(random_parameters['t1']['degrees'])
pprint(random_parameters['t1']['translation'])

dataset.save_sample(transformed, dict(t1='/tmp/t1_motion.nii.gz'))
dataset.save_sample(transformed, dict(label='/tmp/t1_brain_seg_motion.nii.gz'))
