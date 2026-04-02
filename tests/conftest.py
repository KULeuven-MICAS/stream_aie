"""Shared pytest configuration for stream-dse tests."""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (run with pytest -m slow)")
