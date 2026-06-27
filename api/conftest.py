"""Test path setup: make the analysis package importable for the parity test."""

import sys
from pathlib import Path

# api/ is already on pythonpath (pytest.ini). Add analysis/ so the parity test can
# import the source-of-truth feature function and compare against the vendored copy.
ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))
