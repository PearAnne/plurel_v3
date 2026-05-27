from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_rt_sampling_budget import build_diagnostic, load_database_stats_paths


def _write_stats(path: Path, db_name: str, edges: list[dict[str, object]]) -> Path:
    payload = {
        "db_name": db_name,
        "num_tables": 2,
        "num_edges": len(edges),
        "total_child_rows": 10,
        "total_non_null_edges": 10,
        "edges": edges,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_diagnostic_flags_f2p_slot_pressure_and_static_bfs_upper_bound(tmp_path: Path):
    stats_path = _write_stats(
        tmp_path / "edge_topology_stats.rel-synthetic-G0_hsbm-10000.json",
        "rel-synthetic-G0_hsbm-10000",
        [
            {
                "db_name": "rel-synthetic-G0_hsbm-10000",
                "child_table": "child",
                "fkey_col": f"fk_{idx}",
                "parent_table": "parent",
                "num_children": 10,
                "num_parents": 4,
                "num_non_null_edges": 10,
                "metrics": {"fanout_max": 7 if idx == 0 else 3},
                "num_true_null_edges": 0,
                "num_unmatched_fk_edges": 0,
                "is_self_loop": False,
            }
            for idx in range(6)
        ],
    )
    manifest = {
        "cohorts": [
            {
                "cohort": "G0_hsbm",
                "seeds": [10000],
            }
        ]
    }
    pre_root = tmp_path / "pre"
    pre_dir = pre_root / "rel-synthetic-G0_hsbm-10000"
    pre_dir.mkdir(parents=True)
    for name in ("table_info.json", "nodes.rkyv", "offsets.rkyv"):
        (pre_dir / name).write_bytes(b"stub")

    diagnostic = build_diagnostic(
        stats_paths=[stats_path],
        manifest=manifest,
        pre_root=pre_root,
        max_bfs_width=5,
        ctx_len=1024,
    )

    assert diagnostic["limits"]["max_f2p_nbrs"] == 5
    cohort = diagnostic["cohorts"][0]
    assert cohort["cohort"] == "G0_hsbm"
    assert cohort["max_fk_slots"] == 6
    assert cohort["tables_over_f2p_limit"] == 1
    assert cohort["dbs_over_f2p_limit"] == ["rel-synthetic-G0_hsbm-10000"]
    assert cohort["bfs_width_edges_over_budget"] == 1
    assert cohort["ctx_len_status"]["status"] == "unavailable"
    assert "instrumentation" in cohort["ctx_len_status"]["reason"]

    db_pressure = diagnostic["databases"][0]
    assert db_pressure["table_pressures"][0]["status"] == "over_limit"
    assert db_pressure["table_pressures"][0]["fk_slots"] == 6

    bfs_pressure = diagnostic["bfs_width_pressure"]
    assert len(bfs_pressure) == 1
    assert bfs_pressure[0]["status"] == "static_upper_bound_over_budget"
    assert bfs_pressure[0]["fanout_max"] == 7.0

    coverage = diagnostic["preprocessed_file_coverage"][0]
    assert coverage["missing_files"] == ["p2f_adj.rkyv"]
    assert coverage["present_files"] == ["table_info.json", "nodes.rkyv", "offsets.rkyv"]


def test_load_database_stats_paths_discovers_nested_json_files(tmp_path: Path):
    stats_dir = tmp_path / "stats"
    nested = stats_dir / "cohort_a"
    nested.mkdir(parents=True)
    _write_stats(
        nested / "edge_topology_stats.rel-synthetic-G1_erdos_renyi-10001.json",
        "rel-synthetic-G1_erdos_renyi-10001",
        [],
    )

    paths = load_database_stats_paths(stats_paths=[], stats_dirs=[stats_dir])

    assert paths == [
        (nested / "edge_topology_stats.rel-synthetic-G1_erdos_renyi-10001.json").resolve()
    ]
