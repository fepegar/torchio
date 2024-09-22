# pylint: disable=import-outside-toplevel

from pathlib import Path

import typer
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TextColumn

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
    transform_name: str = typer.Argument(...),  # noqa: B008
    output_path: Path = typer.Argument(  # noqa: B008
        ...,
        file_okay=True,
        dir_okay=False,
        writable=True,
    ),
    kwargs: str = typer.Option(  # noqa: B008
        None,
        '--kwargs',
        '-k',
        help='String of kwargs, e.g. "degrees=(-5,15) num_transforms=3".',
    ),
    imclass: str = typer.Option(  # noqa: B008
        'ScalarImage',
        '--imclass',
        '-c',
        help=(
            'Name of the subclass of torchio.Image'
            ' that will be used to instantiate the image.'
        ),
    ),
    seed: int = typer.Option(  # noqa: B008
        None,
        '--seed',
        '-s',
        help='Seed for PyTorch random number generator.',
    ),
    verbose: bool = typer.Option(  # noqa: B008
        False,
        help='Print random transform parameters.',
    ),
    show_progress: bool = typer.Option(  # noqa: B008
        True,
        '--show-progress/--hide-progress',
        '-p/-P',
        help='Show animations indicating progress.',
    ),
):
    """Apply transform to an image.

    Example:
    $ tiotr input.nrrd RandomMotion output.nii "degrees=(-5,15) num_transforms=3" -v
    """
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
    with Progress(
        SpinnerColumn(),
        TextColumn('[progress.description]{task.description}'),
        transient=True,
        disable=not show_progress,
    ) as progress:
        progress.add_task('Applying transform', total=1)
        apply_transform_to_file(
            input_path,
            transform,
            output_path,
            verbose=verbose,
            class_=imclass,
        )


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
    app()
