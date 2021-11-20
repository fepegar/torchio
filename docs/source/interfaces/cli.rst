.. _cli:

Command-line tools
------------------

``tiotr``
^^^^^^^^^

A transform can be quickly applied to an image file using the command-line
tool ``tiotr``, which is automatically installed by ``pip``
during installation of TorchIO::

    $ tiotr input.nii RandomAffine output.nii.gz --kwargs "degrees=(0,0,10) scales=0.1" --seed 42

For more information, run ``tiotr --help``.


``tiohd``
^^^^^^^^^

To print some image metadata, ``tiohd`` can be used. Adding the ``--plot``
argument will plot the image using Matplotlib::

    $ tiohd ~/.cache/torchio/mni_colin27_1998_nifti/colin27_t1_tal_lin.nii
    ScalarImage(shape: (1, 181, 217, 181); spacing: (1.00, 1.00, 1.00); orientation: RAS+; dtype: torch.FloatTensor; memory: 27.1 MiB)

For more information, run ``tiohd --help``.
