import numpy as np

from .data.image import Image, LabelMap
from .data.subject import Subject
from .transforms.preprocessing.spatial.to_canonical import ToCanonical


def import_mpl_plt():
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError('Install matplotlib for plotting support') from e
    return mpl, plt


def rotate(image):
    return np.rot90(image)


def plot_volume(
        image: Image,
        channel=-1,  # default to foreground for binary maps
        axes=None,
        cmap=None,
        output_path=None,
        show=True,
        ):
    _, plt = import_mpl_plt()
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 3)
    image = ToCanonical()(image)
    data = image.data[channel]
    indices = np.array(data.shape) // 2
    i, j, k = indices
    slice_x = rotate(data[i, :, :])
    slice_y = rotate(data[:, j, :])
    slice_z = rotate(data[:, :, k])
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
    x_extent, y_extent, z_extent = [tuple(b) for b in image.bounds.T]
    axes[0].imshow(slice_x, extent=y_extent + z_extent, **kwargs)
    axes[1].imshow(slice_y, extent=x_extent + z_extent, **kwargs)
    axes[2].imshow(slice_z, extent=x_extent + y_extent, **kwargs)
    plt.tight_layout()
    if output_path is not None and fig is not None:
        fig.savefig(output_path)
    if show:
        plt.show()


def plot_subject(
        subject: Subject,
        cmap_dict=None,
        show=True,
        output_path=None,
        figsize=None,
        **kwargs,
        ):
    _, plt = import_mpl_plt()
    fig, axes = plt.subplots(len(subject), 3, figsize=figsize)
    # The array of axes must be 2D so that it can be indexed correctly within
    # the plot_volume() function
    axes = axes.reshape(-1, 3)
    iterable = enumerate(subject.get_images_dict(intensity_only=False).items())
    axes_names = 'sagittal', 'coronal', 'axial'
    for row, (name, image) in iterable:
        row_axes = axes[row]
        cmap = None
        if cmap_dict is not None and name in cmap_dict:
            cmap = cmap_dict[name]
        plot_volume(image, axes=row_axes, show=False, cmap=cmap, **kwargs)
        for axis, axis_name in zip(row_axes, axes_names):
            axis.set_title(f'{name} ({axis_name})')
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path)
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
