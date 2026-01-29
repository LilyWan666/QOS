#!/usr/bin/env python3
"""
Evolve a pairing scoring function with openevolve.

This script uses a file-based setup:
  - evaluation/openevolve_pairing/target.py (evolution target)
  - evaluation/openevolve_pairing/evaluator.py (evaluator)
  - evaluation/openevolve_pairing/config.py (config)
"""

import os
import sys

from openevolve import run_evolution

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVO_DIR = os.path.join(PROJECT_ROOT, "evaluation", "openevolve_pairing")
if EVO_DIR not in sys.path:
    sys.path.append(EVO_DIR)

import config as evo_config
import evaluator as evo_evaluator


def _read_target():
    target_path = os.path.join(EVO_DIR, "target.py")
    with open(target_path, "r", encoding="utf-8") as f:
        return f.read()


def evolve():
    result = run_evolution(
        initial_program=_read_target(),
        evaluator=evo_evaluator.evaluate,
        iterations=evo_config.ITERATIONS,
    )
    print("Best score:", result.best_score)
    print("Best code:\n", result.best_code)


if __name__ == "__main__":
    evolve()
