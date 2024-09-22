#!/usr/bin/env python
"""Tests for CLI tool package."""

from typer.testing import CliRunner

from torchio.cli import apply_transform
from torchio.cli import print_info

from .utils import TorchioTestCase

runner = CliRunner()


class TestCLI(TorchioTestCase):
    def test_cli_transform(self):
        image = str(self.get_image_path('cli'))
        args = [
            image,
            'RandomFlip',
            '--seed',
            '0',
            '--kwargs',
            'axes=(0,1,2)',
            '--hide-progress',
            image,
        ]
        result = runner.invoke(apply_transform.app, args)
        assert result.exit_code == 0
        assert result.output.strip() == ''

    def test_bad_transform(self):
        image = str(self.get_image_path('cli'))
        args = [image, 'RandomRandom', image]
        result = runner.invoke(apply_transform.app, args)
        assert result.exit_code == 1

    def test_cli_hd(self):
        image = str(self.get_image_path('cli'))
        args = [image]
        result = runner.invoke(print_info.app, args)
        assert result.exit_code == 0
        assert (
            result.output == 'ScalarImage('
            'shape: (1, 10, 20, 30);'
            ' spacing: (1.00, 1.00, 1.00);'
            ' orientation: RAS+;'
            ' dtype: torch.DoubleTensor;'
            ' memory: 46.9 KiB'
            ')\n'
        )
