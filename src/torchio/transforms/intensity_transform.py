from ..data.image import Image
from ..data.subject import Subject
from .transform import Transform


class IntensityTransform(Transform):
    """Transform that modifies voxel intensities only."""

    def get_images_dict(self, subject: Subject) -> dict[str, Image]:
        images_dict = subject.get_images_dict(
            intensity_only=True,
            include=self.include,
            exclude=self.exclude,
        )
        return images_dict

    def get_images(self, subject: Subject) -> list[Image]:
        images = subject.get_images(
            intensity_only=True,
            include=self.include,
            exclude=self.exclude,
        )
        return images

    def arguments_are_dict(self) -> bool:
        """Check if main arguments are dict.

        Return ``True`` if the type of all attributes specified in the
        :attr:`args_names` have ``dict`` type.
        """
        args = [getattr(self, name) for name in self.args_names]
        are_dict = [isinstance(arg, dict) for arg in args]
        if all(are_dict):
            return True
        elif not any(are_dict):
            return False
        else:
            message = 'Either all or none of the arguments must be dicts'
            raise ValueError(message)
