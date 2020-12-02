#!/usr/bin/env python

"""Tests for CLI tool package."""

from click.testing import CliRunner
from torchio.cli import apply_transform
from .utils import TorchioTestCase


class TestCLI(TorchioTestCase):
    """Tests for CLI tool."""
    def test_help(self):
        """Test the CLI."""
        runner = CliRunner()
        help_result = runner.invoke(apply_transform.main, ['--help'])
        assert help_result.exit_code == 0
        assert 'Show this message and exit.' in help_result.output

    def test_cli(self):
        image = str(self.get_image_path('cli'))
        runner = CliRunner()
        args = [image, 'RandomFlip', image]
        result = runner.invoke(apply_transform.main, args)
        assert result.exit_code == 0
        assert result.output == ''

    def test_bad_transform(self):
        ValueError
        image = str(self.get_image_path('cli'))
        runner = CliRunner()
        args = [image, 'RandomRandom', image]
        result = runner.invoke(apply_transform.main, args)
        assert result.exit_code == 1
