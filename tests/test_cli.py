#!/usr/bin/env python

"""Tests for CLI tool package."""

from click.testing import CliRunner
from torchio import cli
from .utils import TorchioTestCase


class TestCLI(TorchioTestCase):
    """Tests for CLI tool."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        args = [
            str(self.get_image_path('t1')),
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
