"""
Basic import tests for the Credit Card Fraud Detection project.

These tests verify that all modules can be imported correctly.
"""

import sys
import os
import unittest
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class ImportTests(unittest.TestCase):
    """Test that all modules can be imported correctly."""

    def test_utils_imports(self):
        """Test importing utility modules."""
        from src.utils import config, logger

        self.assertTrue(True)  # If imports succeed, this will run

    def test_data_imports(self):
        """Test importing data modules."""
        from src.data import data_loader

        self.assertTrue(True)  # If imports succeed, this will run

    def test_models_imports(self):
        """Test importing model modules."""
        from src.models import model_selection, model_training

        self.assertTrue(True)  # If imports succeed, this will run

    def test_visualization_imports(self):
        """Test importing visualization modules."""
        from src.visualization import visualization

        self.assertTrue(True)  # If imports succeed, this will run

    def test_main_imports(self):
        """Test importing main modules."""
        from src import main, predict

        self.assertTrue(True)  # If imports succeed, this will run


if __name__ == "__main__":
    unittest.main()
