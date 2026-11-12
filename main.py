"""Run this file to produce comparison.png: every algorithm against the S&P 500.

Works from the VS Code Run button or `python3 main.py`, from any directory.
"""

import os
import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
VENV_PYTHON = FOLDER / ".venv" / "bin" / "python"

# The libraries live in .venv, so switch to its Python if started with another.
if Path(sys.prefix).resolve() != (FOLDER / ".venv").resolve():
    if not VENV_PYTHON.exists():
        raise SystemExit(f"Missing {VENV_PYTHON}; the .venv folder is needed to run this.")
    os.execv(VENV_PYTHON, [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

os.chdir(FOLDER)  # save the PNG next to this file, wherever it was started from

import compare_chart

compare_chart.main()
print(f"PNG file: {FOLDER / 'comparison.png'}")
