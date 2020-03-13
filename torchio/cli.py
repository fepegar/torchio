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
    help='String of kwargs, e.g. "proportion_to_augment=1,num_transforms=3"',
)
@click.option(
    '--seed', '-s',
    type=int,
    help='Seed for PyTorch random number generator',
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
    $ torchio-transform input.nrrd RandomMotion output.nii --kwargs "proportion_to_augment=1 num_transforms=3"
    """
    import torchio.transforms as transforms
    from torchio.transforms.augmentation import RandomTransform
    from torchio.utils import apply_transform_to_file, guess_type

    try:
        transform_class = getattr(transforms, transform_name)
    except AttributeError:
        raise AttributeError(f'"{transform_name}" class not found in torchio')

    params_dict = {}
    if kwargs is not None:
        for substring in kwargs.split():
            key, value_string = substring.split('=')
            value = guess_type(value_string)
            params_dict[key] = value
    if issubclass(transform_class, RandomTransform):
        params_dict['seed'] = seed
    transform = transform_class(**params_dict)
    apply_transform_to_file(
        input_path,
        transform,
        output_path,
    )
    return 0


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    sys.exit(apply_transform())  # pragma: no cover
