History
=======

0.18.0 (29-11-2020)
-------------------

* Add ``FPG`` dataset
* Optimize cropping in samplers
* Optimize implementation of ``UniformSampler`` and ``WeightedSampler``
* Fix non integer labels in Colin 27 version 2008 (#223)
* Add ``RandomAnisotropy`` transform
* Add ``RandomGamma`` transform
* Add ``ScalarImage`` and ``LabelMap`` classes
* Add ``RandomLabelsToImage`` transform
* Add support for more input types in ``Transform``
* Add support for 4D images
* Add ICBM dataset
* Add support to specify axes as anatomical labels
* Add ``SpatialTransform`` and ``IntensityTransform``
* Replace ``ImagesDataset`` with ``SubjectsDataset``
* Add support to pass multiple paths to ``Image``
* Refactor reproducibility features
* Add deterministic versions of all random transforms
* Add support to invert transforms
* Add support for SimpleITK > 1
* Add ``pre-commit`` to help contributions and improve coda quality
* Add DICOM data for testing
* Add some visualization support (``Subject.plot``, ``Image.plot``)
* Add support to pass parameters per axis, e.g. for ``RandomAffine`` (#346)
* Remove deprecated transforms ``Resample`` and ``CenterCropOrPad``

0.17.0 (23-06-2020)
-------------------

* Add transforms history to ``Subject`` attributes to improve traceability
* Add support to use an initial transformation in ``Resample``
* Add support to use an image file as target in ``Resample``
* Add ``mean`` argument to ``RandomNoise``
* Add tensor support for transforms
* Add support to use strings as interpolation argument
* Add support for 2D images
* Add attribute access to ``Subject`` and ``Image``
* Add MNI and 3D Slicer datasets
* Add ``intensity`` argument to ``RandomGhosting``
* Add ``translation`` argument to ``RandomAffine``
* Add shape, spacing and orientation attributes to ``Image`` and ``Subject``
* Refactor samplers
* Refactor inference classes
* Add 3D Slicer extension
* Add ITK-SNAP datasets
* Add support to take NumPy arrays as transforms input
* Optimize cropping using PyTorch
* Optimizing transforms by reducing number of tensor copying
* Improve representation (``repr()``) of ``Image``
* Use lazy loading in ``Image``


0.16.0 (21-04-2020)
-------------------

* Add advanced padding options for ``RandomAffine``
* Add reference space options in ``Resample``
* Add probability argument to all transforms
* Add ``OneOf`` and ``Compose`` transforms to improve composability


0.15.0 (07-04-2020)
-------------------

* Refactor ``RandomElasticDeformation`` transform
* Make ``Subject`` inherit from ``dict``


0.14.0 (31-03-2020)
-------------------

* Add ``datasets`` module
* Add support for DICOM files
* Add documentation
* Add ``CropOrPad`` transform


0.13.0 (24-02-2020)
-------------------

* Add ``Subject`` class
* Add random blur transform
* Add lambda transform
* Add random patches swapping transform
* Add MRI k-space ghosting artefact augmentation


0.12.0 (21-01-2020)
-------------------

* Add ToCanonical transform
* Add CenterCropOrPad transform


0.11.0 (15-01-2020)
-------------------

* Add Resample transform


0.10.0 (15-01-2020)
-------------------

* Add Pad transform
* Add Crop transform


0.9.0 (14-01-2020)
------------------

* Add CLI tool to transform an image from file


0.8.0 (11-01-2020)
------------------

* Add Image class


0.7.0 (02-01-2020)
------------------

* Make transforms use PyTorch tensors consistently


0.6.0 (02-01-2020)
------------------

* Add support for NRRD


0.5.0 (01-01-2020)
------------------

* Add bias field transform


0.4.0 (29-12-2019)
------------------

* Add MRI k-space motion artefact augmentation


0.3.0 (21-12-2019)
------------------

* Add Rescale transform
* Add support for multimodal data and missing modalities


0.2.0 (2019-12-06)
------------------

* First release on PyPI.
