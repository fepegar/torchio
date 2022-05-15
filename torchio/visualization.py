import warnings

import torch
import numpy as np

from .typing import TypePath
from .data.subject import Subject
from .data.image import Image, LabelMap
from .transforms.preprocessing.spatial.to_canonical import ToCanonical
from .transforms.preprocessing.intensity.rescale import RescaleIntensity


def import_mpl_plt():
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError('Install matplotlib for plotting support') from e
    return mpl, plt


def rotate(image, radiological=True, n=-1):
    # Rotate for visualization purposes
    image = np.rot90(image, n)
    if radiological:
        image = np.fliplr(image)
    return image


def plot_volume(
        image: Image,
        radiological=True,
        channel=-1,  # default to foreground for binary maps
        axes=None,
        cmap=None,
        output_path=None,
        show=True,
        xlabels=True,
        percentiles=(0.5, 99.5),
        figsize=None,
        reorient=True,
        indices=None,
        ):
    _, plt = import_mpl_plt()
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    sag_axis, cor_axis, axi_axis = axes

    if reorient:
        image = ToCanonical()(image)
    data = image.data[channel]
    if indices is None:
        indices = np.array(data.shape) // 2
    i, j, k = indices
    slice_x = rotate(data[i, :, :], radiological=radiological)
    slice_y = rotate(data[:, j, :], radiological=radiological)
    slice_z = rotate(data[:, :, k], radiological=radiological)
    kwargs = {}
    is_label = isinstance(image, LabelMap)
    if isinstance(cmap, dict):
        slices = slice_x, slice_y, slice_z
        slice_x, slice_y, slice_z = color_labels(slices, cmap)
    else:
        if cmap is None:
            cmap = 'cubehelix' if is_label else 'gray'
        kwargs['cmap'] = cmap
    if is_label:
        kwargs['interpolation'] = 'none'

    sr, sa, ss = image.spacing
    kwargs['origin'] = 'lower'

    if percentiles is not None and not is_label:
        p1, p2 = np.percentile(data, percentiles)
        kwargs['vmin'] = p1
        kwargs['vmax'] = p2

    sag_aspect = ss / sa
    sag_axis.imshow(slice_x, aspect=sag_aspect, **kwargs)
    if xlabels:
        sag_axis.set_xlabel('A')
    sag_axis.set_ylabel('S')
    sag_axis.invert_xaxis()
    sag_axis.set_title('Sagittal')

    cor_aspect = ss / sr
    cor_axis.imshow(slice_y, aspect=cor_aspect, **kwargs)
    if xlabels:
        cor_axis.set_xlabel('R')
    cor_axis.set_ylabel('S')
    cor_axis.invert_xaxis()
    cor_axis.set_title('Coronal')

    axi_aspect = sa / sr
    axi_axis.imshow(slice_z, aspect=axi_aspect, **kwargs)
    if xlabels:
        axi_axis.set_xlabel('R')
    axi_axis.set_ylabel('A')
    axi_axis.invert_xaxis()
    axi_axis.set_title('Axial')

    plt.tight_layout()
    if output_path is not None and fig is not None:
        fig.savefig(output_path)
    if show:
        plt.show()
    return fig


def plot_subject(
        subject: Subject,
        cmap_dict=None,
        show=True,
        output_path=None,
        figsize=None,
        clear_axes=True,
        **kwargs,
        ):
    _, plt = import_mpl_plt()
    num_images = len(subject)
    many_images = num_images > 2
    subplots_kwargs = {'figsize': figsize}
    try:
        if clear_axes:
            subject.check_consistent_spatial_shape()
            subplots_kwargs['sharex'] = 'row' if many_images else 'col'
            subplots_kwargs['sharey'] = 'row' if many_images else 'col'
    except RuntimeError:  # different shapes in subject
        pass
    args = (3, num_images) if many_images else (num_images, 3)
    fig, axes = plt.subplots(*args, **subplots_kwargs)
    # The array of axes must be 2D so that it can be indexed correctly within
    # the plot_volume() function
    axes = axes.T if many_images else axes.reshape(-1, 3)
    iterable = enumerate(subject.get_images_dict(intensity_only=False).items())
    axes_names = 'sagittal', 'coronal', 'axial'
    for image_index, (name, image) in iterable:
        image_axes = axes[image_index]
        cmap = None
        if cmap_dict is not None and name in cmap_dict:
            cmap = cmap_dict[name]
        last_row = image_index == len(axes) - 1
        plot_volume(
            image,
            axes=image_axes,
            show=False,
            cmap=cmap,
            xlabels=last_row,
            **kwargs,
        )
        for axis, axis_name in zip(image_axes, axes_names):
            axis.set_title(f'{name} ({axis_name})')
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path)
    if show:
        plt.show()


def get_num_bins(x: np.ndarray) -> int:
    """Get the optimal number of bins for a histogram.

    This method uses the Freedman–Diaconis rule to compute the histogram that
    minimizes "the integral of the squared difference between the histogram
    (i.e., relative frequency density) and the density of the theoretical
    probability distribution" (`Wikipedia <https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule>`_).

    Args:
        x: Input values.
    """  # noqa: E501
    # Freedman–Diaconis number of bins
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    bins = round((x.max() - x.min()) / bin_width)
    return bins


def plot_histogram(x: np.ndarray, show=True, **kwargs) -> None:
    _, plt = import_mpl_plt()
    plt.hist(x, bins=get_num_bins(x), **kwargs)
    plt.xlabel('Intensity')
    density = kwargs.pop('density', False)
    ylabel = 'Density' if density else 'Frequency'
    plt.ylabel(ylabel)
    if show:
        plt.show()


def color_labels(arrays, cmap_dict):
    results = []
    for array in arrays:
        si, sj = array.shape
        rgb = np.zeros((si, sj, 3), dtype=np.uint8)
        for label, color in cmap_dict.items():
            if isinstance(color, str):
                mpl, _ = import_mpl_plt()
                color = mpl.colors.to_rgb(color)
                color = [255 * n for n in color]
            rgb[array == label] = color
        results.append(rgb)
    return results


def make_gif(
        tensor: torch.Tensor,
        axis: int,
        duration: float,  # of full gif
        output_path: TypePath,
        loop: int = 0,
        optimize: bool = True,
        rescale: bool = True,
        reverse: bool = False,
        ) -> None:
    try:
        from PIL import Image as ImagePIL
    except ModuleNotFoundError as e:
        message = (
            'Please install Pillow to use Image.to_gif():'
            ' pip install Pillow'
        )
        raise RuntimeError(message) from e
    tensor = RescaleIntensity((0, 255))(tensor) if rescale else tensor
    single_channel = len(tensor) == 1

    # Move channels dimension to the end and bring selected axis to 0
    axes = np.roll(range(1, 4), -axis)
    tensor = tensor.permute(*axes, 0)

    if single_channel:
        mode = 'P'
        tensor = tensor[..., 0]
    else:
        mode = 'RGB'
    array = tensor.byte().numpy()
    n = 2 if axis == 1 else 1
    images = [ImagePIL.fromarray(rotate(i, n=n)).convert(mode) for i in array]
    num_images = len(images)
    images = list(reversed(images)) if reverse else images
    frame_duration_ms = duration / num_images * 1000
    if frame_duration_ms < 10:
        fps = round(1000 / frame_duration_ms)
        frame_duration_ms = 10
        new_duration = frame_duration_ms * num_images / 1000
        message = (
            'The computed frame rate from the given duration is too high'
            f' ({fps} fps). The highest possible frame rate in the GIF'
            ' file format specification is 100 fps. The duration has been set'
            f' to {new_duration:.1f} seconds, instead of {duration:.1f}'
        )
        warnings.warn(message)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        optimize=optimize,
        duration=frame_duration_ms,
        loop=loop,
    )
