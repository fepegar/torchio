# pylint: disable=import-outside-toplevel

"""Console script for torchio."""
import sys
import click


@click.command()
@click.argument('input-path', type=click.Path(exists=True))
@click.option('--plot/--no-plot', '-p', default=False)
@click.option('--show/--no-show', '-s', default=False)
@click.option('--label/--scalar', '-l', default=False)
def main(input_path, plot, show, label):
    """Print information about an image and, optionally, show it.

    \b
    Example:
    $ tiohd input.nii.gz
    """
    # Imports are placed here so that the tool loads faster if not being run
    import torchio as tio
    class_ = tio.LabelMap if label else tio.ScalarImage
    image = class_(input_path)
    image.load()
    print(image)  # noqa: T201
    if plot:
        image.plot()
    if show:
        image.show()
    return 0


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    sys.exit(main())  # pragma: no cover
