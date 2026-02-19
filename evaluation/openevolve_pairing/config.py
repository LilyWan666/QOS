"""Configuration for openevolve pairing search."""
import os
import re


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if raw == "":
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_int_list(name: str, default: str) -> list[int]:
    raw = os.environ.get(name, default)
    values = []
    for tok in re.split(r"[\s,;:|/]+", raw or ""):
        tok = tok.strip()
        if not tok:
            continue
        try:
            values.append(int(tok))
        except Exception:
            continue
    # keep order, de-dup
    out = []
    seen = set()
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _env_str_list(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default)
    values = []
    for tok in re.split(r"[\s,;:|/]+", raw or ""):
        tok = tok.strip().lower()
        if not tok:
            continue
        values.append(tok)
    out = []
    seen = set()
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "")
    if raw == "":
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")

# Pair selection
# TARGET_UTIL = 30
TARGET_UTIL = 60
# TARGET_UTIL = 88
TOP_K = 32
TOP_K_RATIO = _env_float("OE_TOP_K_RATIO", 0.0)
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
EVAL_UTILS = _env_int_list("OE_EVAL_UTILS", str(TARGET_UTIL))
EVAL_SHOTS = _env_int("OE_EVAL_SHOTS", SHOTS)
EVAL_RESTRICT_TO_CSV = _env_bool("OE_RESTRICT_TO_PAIR_CSV", True)

# Evolution
ITERATIONS = 50
EVAL_MODE = "candidate"  # "candidate", "original", "both"

# Objective selection (external switch via env vars):
# - OE_EVAL_OBJECTIVE: avg_rank | corr | combined
# - OE_CORR_METHOD: spearman | pearson
# - OE_CORR_TARGET: inv_rank | neg_rank
# - OE_EVAL_UTILS: utils list, e.g. 30,60,88 (also accepts 30:60:88 / 30 60 88)
# - OE_EVAL_SHOTS: integer shots used for all eval utils
# - OE_RESTRICT_TO_PAIR_CSV: 1/0, evaluate only pairs existing in pair_metrics csv
# - OE_TOP_K_RATIO: Top-K ratio (0.1 or 10 both mean 10%; <=0 disables ratio mode)
# - OE_MULTI_UTIL_AGG: mean | norm_to_baseline
# - OE_PARETO_SECOND_METRIC: fidelity | proxy
# - OE_PROXY_FEATURES: candidate proxy features (comma/space separated)
# - OE_PROXY_FEATURE: one proxy feature to use when OE_PARETO_SECOND_METRIC=proxy
EVAL_OBJECTIVE = os.environ.get("OE_EVAL_OBJECTIVE", "avg_rank").strip().lower()
CORR_METHOD = os.environ.get("OE_CORR_METHOD", "spearman").strip().lower()
CORR_TARGET = os.environ.get("OE_CORR_TARGET", "inv_rank").strip().lower()
CORR_WEIGHT = _env_float("OE_CORR_WEIGHT", 0.7)
AVG_RANK_WEIGHT = _env_float("OE_AVG_RANK_WEIGHT", 0.3)
MULTI_UTIL_AGG = os.environ.get("OE_MULTI_UTIL_AGG", "mean").strip().lower()
PARETO_SECOND_METRIC = os.environ.get("OE_PARETO_SECOND_METRIC", "fidelity").strip().lower()
PROXY_FEATURES = _env_str_list(
    "OE_PROXY_FEATURES",
    "depth_ratio,cnot_ratio,nonlocal_ratio,measure_ratio,instr_ratio",
)
PROXY_FEATURE = os.environ.get(
    "OE_PROXY_FEATURE",
    PROXY_FEATURES[0] if PROXY_FEATURES else "instr_ratio",
).strip().lower()

# Legacy metric weight (kept for compatibility with existing scripts)
FID_WEIGHT = 0.1
