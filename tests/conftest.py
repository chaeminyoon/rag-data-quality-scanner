"""Shared test fixtures and path setup."""

import os
import sys

# Make project root importable (config/, src/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
