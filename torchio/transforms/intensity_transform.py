from .transform import Transform


class IntensityTransform(Transform):
    """Transform that modifies voxel intensities only."""
    @staticmethod
    def get_images_dict(sample):
        return sample.get_images_dict(intensity_only=True)

    def arguments_are_dict(self) -> bool:
        """Check if main arguments are dict.

        Return True if the type of all attributes specified in the
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
