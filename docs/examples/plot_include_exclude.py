"""
Exclude images from transform
=============================

In this example we show how the kwargs ``include`` and ``exclude`` can be
used to apply a transform to only some of the images within a subject.
"""

import torch
import torchio as tio


torch.manual_seed(0)

subject = tio.datasets.Pediatric(years=(4.5, 8.5))
subject.plot()
transform = tio.Compose([
    tio.RandomAffine(degrees=(20, 30)),
    tio.ZNormalization(),
    tio.RandomBlur(std=(3, 4), include='t1'),
    tio.RandomNoise(std=(1, 1.5), exclude='t1'),
])
transformed = transform(subject)
transformed.plot()
