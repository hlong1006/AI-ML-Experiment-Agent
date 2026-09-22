"""
conftest.py — pytest configuration for the entire test suite.
Ensures PYTHONPATH includes the project root.
"""
import sys
from pathlib import Path

# Add project root to sys.path so imports work without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent))
