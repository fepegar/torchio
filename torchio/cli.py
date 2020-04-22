# pylint: disable=import-outside-toplevel

"""Console script for torchio."""
import sys
import click


@click.command()
@click.argument('input-path', type=click.Path(exists=True))
@click.argument('transform-name', type=str)
@click.argument('output-path', type=click.Path())
@click.option(
    '--kwargs', '-k',
    type=str,
    help='String of kwargs, e.g. "degrees=(-5,15) num_transforms=3".',
)
@click.option(
    '--seed', '-s',
    type=int,
    help='Seed for PyTorch random number generator.',
)
def apply_transform(
        input_path,
        transform_name,
        output_path,
        kwargs,
        seed,
        ):
    """Apply transform to an image.

    \b
    Example:
    $ torchio-transform -k "degrees=(-5,15) num_transforms=3" input.nrrd RandomMotion output.nii
    """
    # Imports are placed here so that the tool loads faster if not being run
    import torchio.transforms as transforms
    from torchio.transforms.augmentation import RandomTransform
    from torchio.utils import apply_transform_to_file

    try:
        transform_class = getattr(transforms, transform_name)
    except AttributeError as error:
        message = f'Transform "{transform_name}" not found in torchio'
        raise ValueError(message) from error

    params_dict = get_params_dict_from_kwargs(kwargs)
    if issubclass(transform_class, RandomTransform):
        params_dict['seed'] = seed
    transform = transform_class(**params_dict)
    apply_transform_to_file(
        input_path,
        transform,
        output_path,
    )
    return 0


def get_params_dict_from_kwargs(kwargs):
    from torchio.utils import guess_type
    params_dict = {}
    if kwargs is not None:
        for substring in kwargs.split():
            try:
                key, value_string = substring.split('=')
            except ValueError as error:
                message = f'Arguments string "{kwargs}" not valid'
                raise ValueError(message) from error

            value = guess_type(value_string)
            params_dict[key] = value
    return params_dict


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    sys.exit(apply_transform())  # pragma: no cover
