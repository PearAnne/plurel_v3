"""Default output locations for per-benchmark edge topology statistics."""

from pathlib import Path

TOPOLOGY_STATS_ROOT = Path("out/topology_stats")
RELBENCH_STATS_DIR = TOPOLOGY_STATS_ROOT / "relbench_stats"
CTU_STATS_DIR = TOPOLOGY_STATS_ROOT / "ctu_stats"
DBINFER_STATS_DIR = TOPOLOGY_STATS_ROOT / "dbinfer_stats"

RELBENCH_SUMMARY_PATH = RELBENCH_STATS_DIR / "summary.json"
CTU_SUMMARY_PATH = CTU_STATS_DIR / "summary.json"
DBINFER_SUMMARY_PATH = DBINFER_STATS_DIR / "summary.json"
COMBINED_STATS_DIR = TOPOLOGY_STATS_ROOT / "combined"
COMBINED_SUMMARY_PATH = COMBINED_STATS_DIR / "summary.json"
