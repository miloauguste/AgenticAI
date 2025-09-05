"""
Tests for main module
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import main


def test_main_function_exists():
    """Test that main function exists and is callable"""
    assert callable(main)


def test_main_runs_without_error():
    """Test that main function runs without raising an exception"""
    try:
        main()
    except Exception as e:
        pytest.fail(f"main() raised an exception: {e}")