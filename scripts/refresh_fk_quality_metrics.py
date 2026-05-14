from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from measure_edge_topology import load_relbench_database, parse_db_names

from plurel.topology_adapters import discover_ctu_database_names, load_ctu_database
from plurel.topology_measure import (
    DatabaseTopologyStats,
    EdgeTopologyRecord,
    add_fk_quality_metrics,
    fk_values_to_parent_row_indices_with_quality,
    load_database_stats,
    write_database_stats,
    write_summary,
)
from plurel.topology_stats_paths import CTU_STATS_DIR, RELBENCH_STATS_DIR


def refresh_relbench_stats(
    data_root: Path,
    stats_dir: Path,
    db_names: list[str],
) -> Path:
    output_paths: dict[str, Path] = {}
    refreshed_stats: list[DatabaseTopologyStats] = []

    for db_name in db_names:
        stats = _load_required_stats(stats_dir / f"edge_topology_stats.{db_name}.json")
        database = load_relbench_database(db_name=db_name, data_root=data_root)
        refreshed = _refresh_relbench_database_stats(stats=stats, database=database)
        output_paths[db_name] = write_database_stats(refreshed, stats_dir)
        refreshed_stats.append(refreshed)

    return write_summary(
        db_stats=refreshed_stats,
        output_paths=output_paths,
        data_root=data_root,
        output_dir=stats_dir,
        failures={},
    )


def refresh_ctu_stats(
    data_root: Path,
    stats_dir: Path,
    db_names: list[str] | None,
) -> Path:
    if db_names is None:
        db_names = discover_ctu_database_names(data_root)

    output_paths: dict[str, Path] = {}
    refreshed_stats: list[DatabaseTopologyStats] = []

    for raw_db_name in db_names:
        stats_path = stats_dir / f"edge_topology_stats.ctu-{raw_db_name}.json"
        stats = _load_required_stats(stats_path)
        database = load_ctu_database(ctu_root=data_root, db_name=raw_db_name)
        refreshed = _refresh_topology_database_stats(stats=stats, database=database)
        output_paths[refreshed.db_name] = write_database_stats(refreshed, stats_dir)
        refreshed_stats.append(refreshed)

    return write_summary(
        db_stats=refreshed_stats,
        output_paths=output_paths,
        data_root=data_root,
        output_dir=stats_dir,
        failures={},
    )


def _load_required_stats(stats_path: Path) -> DatabaseTopologyStats:
    stats = load_database_stats(stats_path)
    if stats is None:
        raise FileNotFoundError(f"Could not load existing stats: {stats_path}")
    return stats


def _refresh_relbench_database_stats(
    stats: DatabaseTopologyStats,
    database: Any,
) -> DatabaseTopologyStats:
    refreshed_edges: list[EdgeTopologyRecord] = []
    for edge in stats.edges:
        child_table = database.table_dict[edge.child_table]
        parent_table = database.table_dict[edge.parent_table]
        true_null_mask, unmatched_fk_mask = _relbench_fk_quality_masks(
            series=child_table.df[edge.fkey_col],
            num_parents=len(parent_table.df),
        )
        refreshed_edges.append(
            _with_fk_quality(
                edge=edge,
                true_null_mask=true_null_mask,
                unmatched_fk_mask=unmatched_fk_mask,
            )
        )
    return _with_edges(stats, refreshed_edges)


def _refresh_topology_database_stats(
    stats: DatabaseTopologyStats,
    database: Any,
) -> DatabaseTopologyStats:
    fkeys = {
        (edge.child_table, edge.fkey_col, edge.parent_table): edge for edge in database.foreign_keys
    }

    refreshed_edges: list[EdgeTopologyRecord] = []
    for edge in stats.edges:
        fkey = fkeys[(edge.child_table, edge.fkey_col, edge.parent_table)]
        child_df = database.tables[fkey.child_table].df
        parent_df = database.tables[fkey.parent_table].df
        _, _, true_null_mask, unmatched_fk_mask = fk_values_to_parent_row_indices_with_quality(
            child_df[fkey.fkey_col],
            parent_df,
            fkey.parent_pkey_col,
        )
        refreshed_edges.append(
            _with_fk_quality(
                edge=edge,
                true_null_mask=true_null_mask,
                unmatched_fk_mask=unmatched_fk_mask,
            )
        )
    return _with_edges(stats, refreshed_edges)


def _relbench_fk_quality_masks(
    series: pd.Series, num_parents: int
) -> tuple[np.ndarray, np.ndarray]:
    true_null_mask = series.isna().to_numpy(dtype=bool)
    numeric = pd.to_numeric(series, errors="coerce")
    raw_unmatched = numeric.isna().to_numpy(dtype=bool)
    out_of_range = ((numeric < 0) | (numeric >= num_parents)).fillna(False).to_numpy(dtype=bool)
    unmatched_fk_mask = (raw_unmatched | out_of_range) & ~true_null_mask
    return true_null_mask, unmatched_fk_mask


def _with_fk_quality(
    edge: EdgeTopologyRecord,
    true_null_mask: np.ndarray,
    unmatched_fk_mask: np.ndarray,
) -> EdgeTopologyRecord:
    combined_mask = true_null_mask | unmatched_fk_mask
    metrics = add_fk_quality_metrics(
        metrics=edge.metrics,
        true_null_mask=true_null_mask,
        unmatched_fk_mask=unmatched_fk_mask,
    )
    return replace(
        edge,
        num_non_null_edges=int(np.count_nonzero(~combined_mask)),
        metrics=metrics,
        num_true_null_edges=int(np.count_nonzero(true_null_mask)),
        num_unmatched_fk_edges=int(np.count_nonzero(unmatched_fk_mask)),
    )


def _with_edges(
    stats: DatabaseTopologyStats,
    edges: list[EdgeTopologyRecord],
) -> DatabaseTopologyStats:
    return replace(
        stats,
        total_non_null_edges=sum(edge.num_non_null_edges for edge in edges),
        edges=edges,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refresh FK quality metrics in existing topology stats without recomputing fanout fits."
    )
    parser.add_argument(
        "--family",
        choices=["relbench", "ctu"],
        required=True,
        help="Which existing stats directory to refresh.",
    )
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--stats_dir", type=Path, default=None)
    parser.add_argument(
        "--db_names",
        nargs="+",
        default=None,
        help="Optional db names. CTU names should omit the ctu- prefix.",
    )
    args = parser.parse_args()

    data_root = args.data_root.expanduser()
    stats_dir = (
        args.stats_dir.expanduser()
        if args.stats_dir is not None
        else (RELBENCH_STATS_DIR if args.family == "relbench" else CTU_STATS_DIR)
    )
    db_names = None if args.db_names is None else parse_db_names(args.db_names)

    if args.family == "relbench":
        if db_names is None:
            raise ValueError("--db_names is required for relbench")
        summary_path = refresh_relbench_stats(
            data_root=data_root,
            stats_dir=stats_dir,
            db_names=db_names,
        )
    else:
        summary_path = refresh_ctu_stats(
            data_root=data_root,
            stats_dir=stats_dir,
            db_names=db_names,
        )

    print(f"wrote {summary_path}")
