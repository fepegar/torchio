"""Console script for torchio."""
import sys
import click


@click.command()
@click.argument('input-path', type=click.Path(exists=True))
@click.argument('transform', type=str)
@click.argument('output-path', type=click.Path())
@click.option('--kwargs', '-k', type=str, show_default=True)
@click.option('--seed', '-s', type=int, show_default=True)
@click.option('--verbose/--no-verbose', '-v', default=False, show_default=True)
def apply_transform(input_path, transform, output_path, kwargs, seed, verbose):
    """Apply transform to an image."""
    import torchio.transforms as transforms
    from torchio.utils import apply_transform_to_file
    from torchio.transforms.augmentation import RandomTransform
    if kwargs is not None:  # TODO: implement this
        raise NotImplementedError('Passing kwargs not implemented yet')
    kwargs = {}
    try:
        transform_class = getattr(transforms, transform)
    except AttributeError:
        raise AttributeError(f'"{transform}" class not found in torchio')
    debug_kwargs = dict(verbose=verbose)
    if isinstance(transform_class, RandomTransform):
        debug_kwargs['seed'] = seed
    kwargs.update(debug_kwargs)
    transform = transform_class(**kwargs)
    apply_transform_to_file(
        input_path,
        transform,
        output_path,
    )
    return 0


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    sys.exit(apply_transform())  # pragma: no cover
