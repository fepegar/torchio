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
    '--imclass', '-c',
    type=str,
    default='ScalarImage',
    help='Subclass of torchio.Image used to instantiate the image.'
)
@click.option(
    '--seed', '-s',
    type=int,
    help='Seed for PyTorch random number generator.',
)
@click.option(
    '--verbose/--no-verbose', '-v',
    type=bool,
    default=False,
    help='Print random transform parameters.',
)
def main(
        input_path,
        transform_name,
        output_path,
        kwargs,
        imclass,
        seed,
        verbose,
        ):
    """Apply transform to an image.

    \b
    Example:
    $ torchio-transform -k "degrees=(-5,15) num_transforms=3" input.nrrd RandomMotion output.nii
    """  # noqa: E501
    # Imports are placed here so that the tool loads faster if not being run
    import torch
    import torchio.transforms as transforms
    from torchio.utils import apply_transform_to_file

    try:
        transform_class = getattr(transforms, transform_name)
    except AttributeError as error:
        message = f'Transform "{transform_name}" not found in torchio'
        raise ValueError(message) from error

    params_dict = get_params_dict_from_kwargs(kwargs)
    transform = transform_class(**params_dict)
    if seed is not None:
        torch.manual_seed(seed)
    apply_transform_to_file(
        input_path,
        transform,
        output_path,
        verbose=verbose,
        class_=imclass,
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


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    sys.exit(main())  # pragma: no cover
