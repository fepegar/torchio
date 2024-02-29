from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import SimpleITK as sitk
import torch

from .random_affine import Affine
from .random_affine import RandomAffine
from .random_elastic_deformation import ElasticDeformation
from .random_elastic_deformation import RandomElasticDeformation
from .. import RandomTransform
from ... import SpatialTransform
from ....constants import INTENSITY
from ....constants import TYPE
from ....data.io import nib_to_sitk
from ....data.subject import Subject


class RandomCombinedAffineElasticDeformation(RandomTransform, SpatialTransform):
    r"""Apply a RandomAffine and RandomElasticDeformation simultaneously.

    Optimization to use only a single SimpleITK resampling. For additional details on
    the transformations, see :class:`~torchio.transforms.RandomAffine`
    and :class:`~torchio.transforms.RandomElasticDeformation`

    Args:
        affine_first: Apply affine before elastic deformation.
        affine_kwargs: See :class:`~torchio.transforms.RandomAffine` for kwargs.
        elastic_kwargs: See :class:`~torchio.transforms.RandomElasticDeformation`
            for kwargs.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torchio as tio
        >>> image = tio.datasets.Colin27().t1
        >>> affine_kwargs = {'scales': (0.9, 1.2), 'degrees': 15}
        >>> elastic_kwargs = {'max_displacement': (17, 12, 2)}
        >>> transform = tio.RandomCombinedAffineElasticDeformation(
        ...     affine_kwargs,
        ...     elastic_kwargs
        ... )
        >>> transformed = transform(image)

    .. plot::

        import torchio as tio
        subject = tio.datasets.Slicer('CTChest')
        ct = subject.CT_chest
        elastic_kwargs = {'max_displacement': (17, 12, 2)}
        transform = tio.RandomCombinedAffineElasticDeformation(elastic_kwargs=elastic_kwargs)
        ct_transformed = transform(ct)
        subject.add_image(ct_transformed, 'Transformed')
        subject.plot()
    """

    def __init__(
        self,
        affine_first: bool = True,
        affine_kwargs: Optional[Dict[str, Any]] = None,
        elastic_kwargs: Optional[Dict[str, any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.affine_first = affine_first

        self.affine_kwargs = affine_kwargs or {}
        self.random_affine = RandomAffine(**self.affine_kwargs)

        self.elastic_kwargs = elastic_kwargs or {}
        self.random_elastic = RandomElasticDeformation(**self.elastic_kwargs)

    def get_params(self):
        affine_params = self.random_affine.get_params(
            self.random_affine.scales,
            self.random_affine.degrees,
            self.random_affine.translation,
            self.random_affine.isotropic,
        )
        elastic_params = self.random_elastic.get_params(
            self.random_elastic.num_control_points,
            self.random_elastic.max_displacement,
            self.random_elastic.num_locked_borders,
        )
        return affine_params, elastic_params

    def apply_transform(self, subject: Subject):
        affine_params, elastic_params = self.get_params()

        scaling_params, rotation_params, translation_params = affine_params
        affine_params = {
            'scales': scaling_params.tolist(),
            'degrees': rotation_params.tolist(),
            'translation': translation_params.tolist(),
            'center': self.random_affine.center,
            'default_pad_value': self.random_affine.default_pad_value,
            'image_interpolation': self.random_affine.image_interpolation,
            'label_interpolation': self.random_affine.label_interpolation,
            'check_shape': self.random_affine.check_shape,
        }

        elastic_params = {
            'control_points': elastic_params,
            'max_displacement': self.random_elastic.max_displacement,
            'image_interpolation': self.random_elastic.image_interpolation,
            'label_interpolation': self.random_elastic.label_interpolation,
        }

        arguments = {
            'affine_first': self.affine_first,
            'affine_params': affine_params,
            'elastic_params': elastic_params,
        }

        transform = CombinedAffineElasticDeformation(
            **self.add_include_exclude(arguments)
        )
        transformed = transform(subject)
        assert isinstance(transformed, Subject)
        return transformed


class CombinedAffineElasticDeformation(SpatialTransform):
    r"""Apply an Affine and ElasticDeformation simultaneously.

    Optimization to use only a single SimpleITK resampling. For additional details
    on the transformations, see :class:`~torchio.transforms.augmentation.Affine`
    and :class:`~torchio.transforms.augmentation.ElasticDeformation`

    Args:
        affine_first: Apply affine before elastic deformation.
        affine_kwargs: See :class:`~torchio.transforms.augmentation.RandomAffine` for kwargs.
        elastic_kwargs: See
            :class:`~torchio.transforms.augmentation.RandomElasticDeformation` for kwargs.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(
        self,
        affine_first: bool,
        affine_params: Dict[str, Any],
        elastic_params: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.affine_first = affine_first

        self.affine_params = affine_params
        self._affine = Affine(
            **self.affine_params,
            **kwargs,
        )
        self.elastic_params = elastic_params
        self._elastic = ElasticDeformation(
            **self.elastic_params,
            **kwargs,
        )

        self.args_names = ['affine_first', 'affine_params', 'elastic_params']

    def apply_transform(self, subject: Subject) -> Subject:
        if self._affine.check_shape:
            subject.check_consistent_spatial_shape()
        default_value: float

        for image in self.get_images(subject):
            affine_transform = self._affine.get_affine_transform(image)
            transformed_tensors = []
            for tensor in image.data:
                sitk_image = nib_to_sitk(
                    tensor[np.newaxis],
                    image.affine,
                    force_3d=True,
                )
                if image[TYPE] != INTENSITY:
                    interpolation = self._affine.label_interpolation
                    default_value = 0
                else:
                    interpolation = self._affine.image_interpolation
                    default_value = self._affine.get_default_pad_value(
                        tensor, sitk_image
                    )

                bspline_transform = self._elastic.get_bspline_transform(sitk_image)
                self._elastic.parse_free_form_transform(
                    bspline_transform, self._elastic.max_displacement
                )

                # stack: LIFO
                if self.affine_first:
                    combined_transforms = [affine_transform, bspline_transform]
                else:
                    combined_transforms = [bspline_transform, affine_transform]
                composite_transform = sitk.CompositeTransform(combined_transforms)

                transformed_tensor = self.apply_composite_transform(
                    sitk_image,
                    composite_transform,
                    interpolation,
                    default_value,
                )
                transformed_tensors.append(transformed_tensor)
            image.set_data(torch.stack(transformed_tensors))
        return subject

    def apply_composite_transform(
        self,
        sitk_image: sitk.Image,
        transform: sitk.Transform,
        interpolation: str,
        default_value: float,
    ) -> torch.Tensor:
        floating = reference = sitk_image

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(self.get_sitk_interpolator(interpolation))
        resampler.SetReferenceImage(reference)
        resampler.SetDefaultPixelValue(float(default_value))
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampler.SetTransform(transform)
        resampled = resampler.Execute(floating)

        np_array = sitk.GetArrayFromImage(resampled)
        np_array = np_array.transpose()  # ITK to NumPy
        tensor = torch.as_tensor(np_array)
        return tensor
