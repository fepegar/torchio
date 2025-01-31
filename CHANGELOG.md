# Changelog

## 0.20.0 (2024-09-20)

* Fix silenced exception in Queue when using faulty transform by @Zhack47 in https://github.com/TorchIO-project/torchio/pull/1101
* Fix `RescaleIntensity` by @fepegar in https://github.com/TorchIO-project/torchio/pull/1116
* Queue length modification with the use of DDP by @haughty-yeon in https://github.com/TorchIO-project/torchio/pull/1127
* Fix get_subjects_from_batch ignoring metadata by @KonoMaxi in https://github.com/TorchIO-project/torchio/pull/1131
* Fix random errors in queue test by @fepegar in https://github.com/TorchIO-project/torchio/pull/1142
* Add masking_method arg name in Mask by @lchauvin in https://github.com/TorchIO-project/torchio/pull/1137
* Enable plotting of PNG images with alpha channel by @ChristianHinge in https://github.com/TorchIO-project/torchio/pull/1150
* Fix transforms taking strings instead of sequences by @fepegar in https://github.com/TorchIO-project/torchio/pull/1153
* Improve docs for "isotropic" arg in RandomAffine by @AminAlam in https://github.com/TorchIO-project/torchio/pull/1163
* Add support to slice images and subjects by @fepegar in https://github.com/TorchIO-project/torchio/pull/1170
* Make transform return the same type as input by @haarisr in https://github.com/TorchIO-project/torchio/pull/1182
* Fix error which copying a subclass of Subject with keyword attributes by @c-winder in https://github.com/TorchIO-project/torchio/pull/1186
* Add SubjectsLoader by @fepegar in https://github.com/TorchIO-project/torchio/pull/1194

## 0.19.0 (2023-07-26)

* Replace p by **kwargs in all transforms by @dmus in https://github.com/TorchIO-project/torchio/pull/370
* Implementation of include/exclude args by @dmus in https://github.com/TorchIO-project/torchio/pull/373
* Remove internal self.fill, and keep padding_mode as string or number by @romainVala in https://github.com/TorchIO-project/torchio/pull/378
* Remove checks and casts to float32 by @dmus in https://github.com/TorchIO-project/torchio/pull/380
* Fix conversion to Pillow image by @fepegar in https://github.com/TorchIO-project/torchio/pull/385
* Print image path if probability map is empty by @fepegar in https://github.com/TorchIO-project/torchio/pull/386
* Remove torchvision requirement by @fepegar in https://github.com/TorchIO-project/torchio/pull/391
* Fix label sampler and improve error message by @fepegar in https://github.com/TorchIO-project/torchio/pull/394
* Fix error with invertible custom transforms, also keep include/exclude lists for inverse transforms. by @efirdc in https://github.com/TorchIO-project/torchio/pull/398
* New label transformations by @efirdc in https://github.com/TorchIO-project/torchio/pull/402
* Add transform to ensure shape multiple of N by @fepegar in https://github.com/TorchIO-project/torchio/pull/401
* Allow empty list as input to Compose. by @efirdc in https://github.com/TorchIO-project/torchio/pull/404
* Cast to float before computing mean by @fepegar in https://github.com/TorchIO-project/torchio/pull/408
* Add option to keep copies of original images by @fepegar in https://github.com/TorchIO-project/torchio/pull/413
* Delete image attribute when removing image by @fepegar in https://github.com/TorchIO-project/torchio/pull/415
* Add option to ignore intensity transforms when inverting by @fepegar in https://github.com/TorchIO-project/torchio/pull/417
* Add OneHot transform by @fepegar in https://github.com/TorchIO-project/torchio/pull/420
* Add option to modify interpolation when inverting by @fepegar in https://github.com/TorchIO-project/torchio/pull/418
* Add some label transforms for binary morphological operations by @fepegar in https://github.com/TorchIO-project/torchio/pull/424
* Small improvements for Queue by @fepegar in https://github.com/TorchIO-project/torchio/pull/423
* Fix CropOrPad for 2D images by @fepegar in https://github.com/TorchIO-project/torchio/pull/435
* Add EPISURG dataset by @fepegar in https://github.com/TorchIO-project/torchio/pull/433
* Add list of masks functionality to HistogramStandardization.train by @Linardos in https://github.com/TorchIO-project/torchio/pull/446
* Add support for custom image readers by @fepegar in https://github.com/TorchIO-project/torchio/pull/454
* Fix probabilities in label sampler by @fepegar in https://github.com/TorchIO-project/torchio/pull/459
* Use voxels rather than bounds when plotting images by @fepegar in https://github.com/TorchIO-project/torchio/pull/487
* Add input range kwarg for RescaleIntensity by @fepegar in https://github.com/TorchIO-project/torchio/pull/499
* Ignore NumPy underflows by @fepegar in https://github.com/TorchIO-project/torchio/pull/503
* Remove unused imports by @deepsource-autofix in https://github.com/TorchIO-project/torchio/pull/506
* Remove methods with unnecessary super delegation. by @deepsource-autofix in https://github.com/TorchIO-project/torchio/pull/507
* Add examples gallery to documentation by @fepegar in https://github.com/TorchIO-project/torchio/pull/508
* Fix 0 prob random sample in WeightedSampler by @TylerSpears in https://github.com/TorchIO-project/torchio/pull/511
* Avoid deprecation warnings by @fepegar in https://github.com/TorchIO-project/torchio/pull/517
* Add functions to process batches by @fepegar in https://github.com/TorchIO-project/torchio/pull/516
* Make the grid sampler compatible with the queue by @fepegar in https://github.com/TorchIO-project/torchio/pull/520
* Compress and cleanup downloaded datasets by @fepegar in https://github.com/TorchIO-project/torchio/pull/531
* Set minimum version for NumPy requirement by @fepegar in https://github.com/TorchIO-project/torchio/pull/532
* Fix unexpected exception handling by @siahuat0727 in https://github.com/TorchIO-project/torchio/pull/542
* Fix 'unraveling -1' in the weighted random sampler. by @dvolgyes in https://github.com/TorchIO-project/torchio/pull/551
* Add Mask transform by @Svdvoort in https://github.com/TorchIO-project/torchio/pull/527
* Add CopyAffine transform by @albansteff in https://github.com/TorchIO-project/torchio/pull/584
* Added Image method to convert from sitk to torchio by @mattwarkentin in https://github.com/TorchIO-project/torchio/pull/616
* Added intensity clamp transform by @mattwarkentin in https://github.com/TorchIO-project/torchio/pull/622
* Fix default pad value for label maps by @fepegar in https://github.com/TorchIO-project/torchio/pull/627
* Add tolerance for numeric checks by @justusschock in https://github.com/TorchIO-project/torchio/pull/592
* Fix syntax error for Python < 3.8 by @fepegar in https://github.com/TorchIO-project/torchio/pull/639
* Add support to pass shape and affine to Resample by @fepegar in https://github.com/TorchIO-project/torchio/pull/640
* Fix single slices metadata being lost by @fepegar in https://github.com/TorchIO-project/torchio/pull/641
* Add Resize transform by @fepegar in https://github.com/TorchIO-project/torchio/pull/642
* Raise from None to reduce error verbosity by @fepegar in https://github.com/TorchIO-project/torchio/pull/649
* Add RSNA-MICCAI brain tumor dataset by @fepegar in https://github.com/TorchIO-project/torchio/pull/650
* Fix visualization for subject with many images by @fepegar in https://github.com/TorchIO-project/torchio/pull/653
* Fix error when normalization mask is empty by @fepegar in https://github.com/TorchIO-project/torchio/pull/656
* Enable Indexing by int-Compatible type by @justusschock in https://github.com/TorchIO-project/torchio/pull/670
* Add support to crop/pad to the mask's bounding box by @fepegar in https://github.com/TorchIO-project/torchio/pull/678
* Add support to save GIFs by @fepegar in https://github.com/TorchIO-project/torchio/pull/680
* Add some Visible Human datasets by @fepegar in https://github.com/TorchIO-project/torchio/pull/679
* Add indices parameter for plotting volumes by @laynr in https://github.com/TorchIO-project/torchio/pull/683
* Remove channels_last kwarg from Image by @fepegar in https://github.com/TorchIO-project/torchio/pull/685
* Add parse_input kwarg to Transform by @fepegar in https://github.com/TorchIO-project/torchio/pull/696
* Cast translation argument to float64 for SimpleITK by @fepegar in https://github.com/TorchIO-project/torchio/pull/702
* Make Affine take RAS parameters of floating->ref by @fepegar in https://github.com/TorchIO-project/torchio/pull/712
* Rename master to main by @fepegar in https://github.com/TorchIO-project/torchio/pull/717
* Add method to show image with external viewer by @fepegar in https://github.com/TorchIO-project/torchio/pull/727
* Add kwarg to not check spatial consistency by @fepegar in https://github.com/TorchIO-project/torchio/pull/735
* Retrieve transforms history from batch dictionary by @fepegar in https://github.com/TorchIO-project/torchio/pull/745
* Add logic to invert OneHot transform by @fepegar in https://github.com/TorchIO-project/torchio/pull/748
* Fix wrong class when creating subjects from batch by @fepegar in https://github.com/TorchIO-project/torchio/pull/749
* Minor quality of life functions for inspecting scalar images and label maps by @mattwarkentin in https://github.com/TorchIO-project/torchio/pull/728
* Fix CropOrPad incorrectly storing bounds by @fepegar in https://github.com/TorchIO-project/torchio/pull/760
* Fix affine getter for multipath images by @fepegar in https://github.com/TorchIO-project/torchio/pull/763
* Check data type compatibility for custom reader by @fepegar in https://github.com/TorchIO-project/torchio/pull/770
* Use PyTorch to shuffle the queue patches by @fepegar in https://github.com/TorchIO-project/torchio/pull/776
* Fix `Blur` using only the first standard deviation by @fepegar in https://github.com/TorchIO-project/torchio/pull/772
* Add support to specify the interpolation type for label images by @snavalm in https://github.com/TorchIO-project/torchio/pull/791
* Fix overlapping in patches aggregator by @fepegar in https://github.com/TorchIO-project/torchio/pull/832
* Add support for torch >= 1.11 by @snipdome in https://github.com/TorchIO-project/torchio/pull/838
* Fix new SimpleITK release not being installed by @fepegar in https://github.com/TorchIO-project/torchio/pull/852
* Fix the wrong mapping in sequential labels transform by @iamSmallY in https://github.com/TorchIO-project/torchio/pull/841
* Return figure in plot_volume function by @dmus in https://github.com/TorchIO-project/torchio/pull/872
* Use PyTorch to compute Fourier transforms by @fepegar in https://github.com/TorchIO-project/torchio/pull/389
* Remove support for Python 3.6 by @fepegar in https://github.com/TorchIO-project/torchio/pull/881
* Add support to use different numbers of samples in the queue by @fepegar in https://github.com/TorchIO-project/torchio/pull/795
* Fix computation of kernels in `Blur` transform by @fepegar in https://github.com/TorchIO-project/torchio/pull/861
* Add support to pass label keys for dict input by @fepegar in https://github.com/TorchIO-project/torchio/pull/879
* Add support to mask 4D images with 3D masks by @fepegar in https://github.com/TorchIO-project/torchio/pull/908
* Add fftshift to fourier_transform with torch.fft by @iimog in https://github.com/TorchIO-project/torchio/pull/912
* Fix copying of subclasses of Subject by @justusschock in https://github.com/TorchIO-project/torchio/pull/794
* Use tqdm.auto by @fepegar in https://github.com/TorchIO-project/torchio/pull/926
* Add hann window function in GridAggregator as padding_mode by @LucaLumetti in https://github.com/TorchIO-project/torchio/pull/900
* Add RSNA 2022 Cervical Spine Fracture Detection dataset by @fepegar in https://github.com/TorchIO-project/torchio/pull/943
* Replace Click with Typer by @fepegar in https://github.com/TorchIO-project/torchio/pull/978
* Fix number of samples per subject in queue by @fepegar in https://github.com/TorchIO-project/torchio/pull/981
* Stop loading data when copying unloaded image by @fepegar in https://github.com/TorchIO-project/torchio/pull/982
* Add support to pass a callable as padding mode by @mueller-franzes in https://github.com/TorchIO-project/torchio/pull/959
* Use indexing instead of masked_select for faster normalization by @ramonemiliani93 in https://github.com/TorchIO-project/torchio/pull/1018
* Remove Visible Human Project datasets by @fepegar in https://github.com/TorchIO-project/torchio/pull/1026
* Add support for Python 3.11 by @fepegar in https://github.com/TorchIO-project/torchio/pull/1008
* Make subject parameter mandatory by @fepegar in https://github.com/TorchIO-project/torchio/pull/1029
* Improve support for queue in distributed training by @hsyang1222 in https://github.com/TorchIO-project/torchio/pull/1021
* Stop ignoring intensity transforms when computing inverse by @fepegar in https://github.com/TorchIO-project/torchio/pull/1039
* Add support to invert RescaleIntensity transform by @nicoloesch in https://github.com/TorchIO-project/torchio/pull/998
* Fix HistogramStandardization example by @vedal in https://github.com/TorchIO-project/torchio/pull/1022
* Add workflow to validate pull request title by @fepegar in https://github.com/TorchIO-project/torchio/pull/1042
* Add support to unload images by @fepegar in https://github.com/TorchIO-project/torchio/pull/983
* Split CI workflows by @fepegar in https://github.com/TorchIO-project/torchio/pull/1047
* Warn if 'mean' padding mode is used for label maps by @fepegar in https://github.com/TorchIO-project/torchio/pull/1065
* Fix unloaded image affine not read after CopyAffine by @fepegar in https://github.com/TorchIO-project/torchio/pull/1072
* Fix custom reader not propagated when copying by @fepegar in https://github.com/TorchIO-project/torchio/pull/1091
* Remove support for Python 3.7 by @fepegar in https://github.com/TorchIO-project/torchio/pull/1099

## 0.18.0 (2020-11-29)

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
* Add ``pre-commit`` to help contributions and improve code quality
* Add DICOM data for testing
* Add some visualization support (``Subject.plot``, ``Image.plot``)
* Add support to pass parameters per axis, e.g. for ``RandomAffine`` (#346)
* Remove deprecated transforms ``Resample`` and ``CenterCropOrPad``

## 0.17.0 (2020-06-23)

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

## 0.16.0 (2020-04-21)

* Add advanced padding options for ``RandomAffine``
* Add reference space options in ``Resample``
* Add probability argument to all transforms
* Add ``OneOf`` and ``Compose`` transforms to improve composability

## 0.15.0 (2020-04-07)

* Refactor ``RandomElasticDeformation`` transform
* Make ``Subject`` inherit from ``dict``

## 0.14.0 (2020-03-31)

* Add ``datasets`` module
* Add support for DICOM files
* Add documentation
* Add ``CropOrPad`` transform

## 0.13.0 (2020-02-24)

* Add ``Subject`` class
* Add random blur transform
* Add lambda transform
* Add random patches swapping transform
* Add MRI k-space ghosting artefact augmentation

## 0.12.0 (2020-01-21)

* Add ToCanonical transform
* Add CenterCropOrPad transform

## 0.11.0 (2020-01-15)

* Add Resample transform

## 0.10.0 (2020-01-15)

* Add Pad transform
* Add Crop transform

## 0.9.0 (2020-01-14)

* Add CLI tool to transform an image from file

## 0.8.0 (2020-01-11)

* Add Image class

## 0.7.0 (2020-01-02)

* Make transforms use PyTorch tensors consistently

## 0.6.0 (2020-01-02)

* Add support for NRRD

## 0.5.0 (2020-01-01)

* Add bias field transform

## 0.4.0 (2019-12-29)

* Add MRI k-space motion artefact augmentation

## 0.3.0 (2019-12-21)

* Add Rescale transform
* Add support for multimodal data and missing modalities

## 0.2.0 (2019-12-06)

* First release on PyPI.
