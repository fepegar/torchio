# pylint: disable=import-outside-toplevel

"""Console script for torchio."""
import sys
import click


@click.command()
@click.argument('input-path', type=click.Path(exists=True))
def main(input_path):
    """Print information about an image.

    \b
    Example:
    $ tiohd input.nii.gz
    """
    # Imports are placed here so that the tool loads faster if not being run
    import torchio as tio
    image = tio.ScalarImage(input_path)
    image.load()
    print(image)  # noqa: T001
    return 0


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    sys.exit(main())  # pragma: no cover
