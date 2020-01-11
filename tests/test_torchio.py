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
        nii = nib.Nifti1Image(np.empty((10, 20, 30)), np.eye(4))
        nii.to_filename(str(self.path))

    def tearDown(self):
        """Tear down test fixtures, if any."""
        self.path.unlink()

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        args = [str(self.path), 'ZNormalization', '/tmp/test.nii']
        result = runner.invoke(cli.apply_transform, args)
        assert result.exit_code == 0
        help_result = runner.invoke(cli.apply_transform, ['--help'])
        assert help_result.exit_code == 0
        assert 'Show this message and exit.' in help_result.output
