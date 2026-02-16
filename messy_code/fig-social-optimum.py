"""Wrapper to run the canonical figure script from `analysis/`."""

from pathlib import Path
import runpy

SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "fig_user_equilibrium_vs_social_optimum.py"
runpy.run_path(SCRIPT, run_name="__main__")
