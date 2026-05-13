from __future__ import annotations

import json
from pathlib import Path

from plurel.topology_measure import DatabaseTopologyStats, EdgeTopologyRecord, write_database_stats
from scripts.merge_topology_summaries import merge_topology_stats_dirs


def _write_benchmark_dir(
    root: Path,
    benchmark_name: str,
    db_name: str,
    *,
    gamma: float,
) -> Path:
    benchmark_dir = root / benchmark_name
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    stats = DatabaseTopologyStats(
        db_name=db_name,
        num_tables=1,
        num_edges=1,
        total_child_rows=10,
        total_non_null_edges=10,
        edges=[
            EdgeTopologyRecord(
                db_name=db_name,
                child_table="child",
                fkey_col="fk",
                parent_table="parent",
                num_children=10,
                num_parents=5,
                num_non_null_edges=10,
                metrics={"powerlaw_gamma": gamma, "powerlaw_plausible": True},
            )
        ],
    )
    stats_path = write_database_stats(stats, benchmark_dir)
    summary = {
        "data_root": f"/data/{benchmark_name}",
        "output_dir": str(benchmark_dir),
        "num_dbs": 1,
        "db_names": [db_name],
        "rows": [],
        "dbs": {
            db_name: {
                "stats_path": str(stats_path),
                "num_tables": 1,
                "num_edges": 1,
                "total_child_rows": 10,
                "total_non_null_edges": 10,
            }
        },
        "metric_summary": {},
        "metric_summary_plausible": {},
        "failures": {},
    }
    (benchmark_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return benchmark_dir


def test_merge_topology_stats_dirs_combines_rows_and_metric_summaries(tmp_path: Path):
    relbench_dir = _write_benchmark_dir(tmp_path, "relbench_stats", "rel-f1", gamma=2.0)
    ctu_dir = _write_benchmark_dir(tmp_path, "ctu_stats", "ctu-financial", gamma=3.0)
    output_dir = tmp_path / "combined"

    summary_path = merge_topology_stats_dirs(
        input_dirs=[relbench_dir, ctu_dir],
        output_dir=output_dir,
    )

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["num_dbs"] == 2
    assert {row["db_name"] for row in payload["rows"]} == {"rel-f1", "ctu-financial"}
    assert payload["metric_summary"]["powerlaw_gamma"]["count"] == 2
