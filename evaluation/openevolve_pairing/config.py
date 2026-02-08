"""Configuration for openevolve pairing search."""
import os


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)

# Pair selection
# TARGET_UTIL = 30
TARGET_UTIL = 60
# TARGET_UTIL = 88
TOP_K = 24
# Always evaluate on the full candidate pool.
CANDIDATE_LIMIT = None

# SLA thresholds (normalized)
SLA_MIN_UTIL = 0.0
SLA_MIN_FID = 0.0

# Simulation
LAYOUT_MODE = "qos"  # "qos" or "baseline"
BASELINE_USE_SCHEDULER = False
SHOTS = 8196
RANDOM_SEED = 42

# Evolution
ITERATIONS = 50
EVAL_MODE = "candidate"  # "candidate", "original", "both"

# Objective selection (external switch via env vars):
# - OE_EVAL_OBJECTIVE: avg_rank | corr | combined
# - OE_CORR_METHOD: spearman | pearson
# - OE_CORR_TARGET: inv_rank | neg_rank
EVAL_OBJECTIVE = os.environ.get("OE_EVAL_OBJECTIVE", "avg_rank").strip().lower()
CORR_METHOD = os.environ.get("OE_CORR_METHOD", "spearman").strip().lower()
CORR_TARGET = os.environ.get("OE_CORR_TARGET", "inv_rank").strip().lower()
CORR_WEIGHT = _env_float("OE_CORR_WEIGHT", 0.7)
AVG_RANK_WEIGHT = _env_float("OE_AVG_RANK_WEIGHT", 0.3)

# Legacy metric weight (kept for compatibility with existing scripts)
FID_WEIGHT = 0.1
