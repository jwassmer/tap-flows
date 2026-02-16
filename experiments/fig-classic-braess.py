"""Wrapper to run the canonical figure script from `analysis/`."""

from pathlib import Path
import runpy

SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "fig_classic_braess_social_cost.py"
runpy.run_path(SCRIPT, run_name="__main__")
