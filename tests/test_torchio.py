#!/usr/bin/env python

"""Tests for `torchio` package."""

import unittest
import tempfile
from pathlib import Path
from click.testing import CliRunner
import numpy as np
import nibabel as nib
from torchio import cli


class TestTorchio(unittest.TestCase):
    """Tests for `torchio` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.path = Path(tempfile.gettempdir()) / 'test.nii'
        nii = nib.Nifti1Image(np.random.rand(10, 20, 30), np.eye(4))
        nii.to_filename(str(self.path))

    def tearDown(self):
        """Tear down test fixtures, if any."""
        self.path.unlink()

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        args = [
            str(self.path),
            'Rescale',
            '/tmp/test.nii',
            # '-k', r'"out_min_max=(0,1) percentiles=(10,90)"',  # not working
            '-k', 'out_min_max=(-1,1)',
        ]
        result = runner.invoke(cli.apply_transform, args)
        assert result.exit_code == 0
        help_result = runner.invoke(cli.apply_transform, ['--help'])
        assert help_result.exit_code == 0
        assert 'Show this message and exit.' in help_result.output

        # # Why doesn't this work?
        # args = [str(self.path), 'NoExist', '/tmp/test.nii']
        # with self.assertRaises(AttributeError):
        #     result = runner.invoke(cli.apply_transform, args)
