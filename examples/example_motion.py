from pprint import pprint
from torchio import Image, ImagesDataset, transforms, INTENSITY, LABEL, Subject

subject = Subject(
    Image('label', '~/Dropbox/MRI/t1_brain_seg.nii.gz', LABEL),
    Image('t1', '~/Dropbox/MRI/t1.nii.gz', INTENSITY),
)
subjects_list = [subject]

dataset = ImagesDataset(subjects_list)
sample = dataset[0]
transform = transforms.RandomMotion(
    seed=42,
    degrees=10,
    translation=10,
    num_transforms=3,
    verbose=True,
    proportion_to_augment=1,
)
transformed = transform(sample)

pprint(transformed['t1']['random_motion_times'])
pprint(transformed['t1']['random_motion_degrees'])
pprint(transformed['t1']['random_motion_translation'])

dataset.save_sample(transformed, dict(t1='/tmp/t1_motion.nii.gz'))
dataset.save_sample(transformed, dict(label='/tmp/t1_brain_seg_motion.nii.gz'))
