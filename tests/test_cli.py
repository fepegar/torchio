#!/usr/bin/env python

"""Tests for CLI tool package."""

from click.testing import CliRunner
from torchio import cli
from .utils import TorchioTestCase


class TestCLI(TorchioTestCase):
    """Tests for CLI tool."""
    def test_help(self):
        """Test the CLI."""
        runner = CliRunner()
        help_result = runner.invoke(cli.apply_transform, ['--help'])
        assert help_result.exit_code == 0
        assert 'Show this message and exit.' in help_result.output

    def test_wrong_transform(self):
        runner = CliRunner()
        args = [str(self.get_image_path('image')), 'NoExist', '/tmp/test.nii']
        print(' '.join(args))
        result = runner.invoke(cli.apply_transform, args)
        assert result.exit_code == 1

    def test_wrong_kwargs(self):
        runner = CliRunner()
        args = [
            str(self.get_image_path('image')),
            'Rescale',
            '/tmp/test.nii',
            '-k', '"out_min_max=(0,1),percentiles=(10,90)"'
        ]
        result = runner.invoke(cli.apply_transform, args)
        assert result.exit_code == 1

    def test_valid_kwargs(self):
        runner = CliRunner()
        args = [
            str(self.get_image_path('image')),
            'Rescale',
            '/tmp/test.nii',
            '-k', '"out_min_max=(0,1) percentiles=(10,90)"'
        ]
        result = runner.invoke(cli.apply_transform, args)
        assert result.exit_code == 0
