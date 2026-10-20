#!/usr/bin/env python3
"""Run the end-to-end experimentation-platform demo (offline).

Shows three scenarios against known synthetic ground truth: a real
lift that ships, a dropped-exposure SRM bug that invalidates the
read, and a guardrail regression that vetoes shipping.

    PYTHONPATH=src python examples/run_demo.py
    # equivalently: experimentation-platform demo
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experimentation_platform.demo import run_demo

if __name__ == "__main__":
    run_demo()
