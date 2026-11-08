"""Convenience entry point; run after installation or with PYTHONPATH=src."""

import sys

from furniture_bench.cli import main

raise SystemExit(main(["visualize", *sys.argv[1:]]))
