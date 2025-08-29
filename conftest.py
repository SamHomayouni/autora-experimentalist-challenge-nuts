"""Pytest configuration for ensuring the `src` directory is on ``sys.path``.

This allows the test suite to import the package without installing it.
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

