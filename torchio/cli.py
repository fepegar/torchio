"""Console script for torchio."""
import sys
import click


@click.command()
@click.argument('input-path', type=click.Path(exists=True))
@click.argument('transform-name', type=str)
@click.argument('output-path', type=click.Path())
@click.option('--kwargs', '-k', type=str, help='String of kwargs, e.g. "proportion_to_augment=1,num_transforms=3"')
@click.option('--seed', '-s', type=int, help='Seed for PyTorch random number generator')
@click.option('--verbose/--no-verbose', '-v', default=False, show_default=True)
def apply_transform(input_path, transform_name, output_path, kwargs, seed, verbose):
    """Apply transform to an image.

    \b
    Example:
    $ torchio-transform input.nrrd RandomMotion output.nii --kwargs "proportion_to_augment=1 num_transforms=3"
    """
    import torchio.transforms as transforms
    from torchio.utils import apply_transform_to_file
    from torchio.transforms.augmentation import RandomTransform

    try:
        transform_class = getattr(transforms, transform_name)
    except AttributeError:
        raise AttributeError(f'"{transform_name}" class not found in torchio')

    params_dict = {}
    if kwargs is not None:
        for substring in kwargs.split():
            key, value_string = substring.split('=')
            value_type = guess_type(value_string)
            try:
                value = value_type(value_string)
            except TypeError:
                value = None
            params_dict[key] = value

    debug_kwargs = dict(verbose=verbose)
    if isinstance(transform_class, RandomTransform):
        debug_kwargs['seed'] = seed
    params_dict.update(debug_kwargs)
    transform = transform_class(**params_dict)
    apply_transform_to_file(
        input_path,
        transform,
        output_path,
    )
    return 0


def guess_type(variable):
    """
    Adapted from
    https://www.reddit.com/r/learnpython/comments/4599hl/module_to_guess_type_from_a_string/czw3f5s
    """
    import ast
    try:
        value = ast.literal_eval(variable)
    except ValueError:
        return str
    else:
        return type(value)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    sys.exit(apply_transform())  # pragma: no cover
