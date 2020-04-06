from torchio.transforms import Lambda
from torchio import Image, ImagesDataset, INTENSITY, LABEL, Subject

subject = Subject(
    label=Image('~/Dropbox/MRI/t1_brain_seg.nii.gz', LABEL),
    t1=Image('~/Dropbox/MRI/t1.nii.gz', INTENSITY),
)
subjects_list = [subject]

dataset = ImagesDataset(subjects_list)
sample = dataset[0]
transform = Lambda(lambda x: -1.5 * x, types_to_apply=INTENSITY)
transformed = transform(sample)
dataset.save_sample(transformed, {'t1': '/tmp/t1_lambda.nii'})
