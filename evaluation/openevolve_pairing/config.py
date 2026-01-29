"""Configuration for openevolve pairing search."""

# Pair selection
# TARGET_UTIL = 30
TARGET_UTIL = 60
# TARGET_UTIL = 88
TOP_K = 24
CANDIDATE_LIMIT = 120

# SLA thresholds (normalized)
SLA_MIN_UTIL = 0.0
SLA_MIN_FID = 0.0

# Simulation
LAYOUT_MODE = "qos"  # "qos" or "baseline"
BASELINE_USE_SCHEDULER = False
SHOTS = 1000
RANDOM_SEED = 42

# Evolution
ITERATIONS = 50
EVAL_MODE = "candidate"  # "candidate", "original", "both"

# Evaluation metric: score = 1/avg_rank + FID_WEIGHT * avg_fid (Top-K)
FID_WEIGHT = 0.1
