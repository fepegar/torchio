# pylint: disable=import-outside-toplevel
from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def main(
    input_path: Path = typer.Argument(  # noqa: B008
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
    ),
    plot: bool = typer.Option(  # noqa: B008
        False,
        '--plot/--no-plot',
        '-p/-P',
        help='Plot the image using Matplotlib or Pillow.',
    ),
    show: bool = typer.Option(  # noqa: B008
        False,
        '--show/--no-show',
        '-s/-S',
        help='Show the image using specialized visualisation software.',
    ),
    label: bool = typer.Option(  # noqa: B008
        False,
        '--label/--scalar',
        '-l/-s',
        help='Use torchio.LabelMap to instantiate the image.',
    ),
):
    """Print information about an image and, optionally, show it.

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


if __name__ == '__main__':
    app()
