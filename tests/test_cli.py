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
