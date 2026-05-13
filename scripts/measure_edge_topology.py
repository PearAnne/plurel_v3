from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from relbench.base import Database

from plurel.topology_measure import (
    DatabaseTopologyStats,
    EdgeTopologyRecord,
    load_database_stats,
    summarize_metrics,
    write_database_stats,
    write_summary,
)
from plurel.topology_metrics import DEFAULT_MAX_POWERLAW_SAMPLE


def load_relbench_database(db_name: str, data_root: Path) -> Database:
    from relbench.base import Database
    from relbench.datasets import dataset_registry

    db_cache_dir = data_root.expanduser() / db_name
    db_dir = db_cache_dir / "db"

    if _contains_parquet(db_dir):
        return Database.load(db_dir)

    if _contains_parquet(db_cache_dir):
        return Database.load(db_cache_dir)

    if db_name not in dataset_registry:
        available = ", ".join(sorted(dataset_registry))
        raise KeyError(f"Unknown RelBench dataset '{db_name}'. Available: {available}")

    cls, args, kwargs = dataset_registry[db_name]
    dataset = cls(*args, **{**kwargs, "cache_dir": str(db_cache_dir)})
    return dataset.get_db(upto_test_timestamp=False)


def _contains_parquet(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.parquet"))


def measure_database(
    db_name: str,
    db: Database,
    max_powerlaw_sample: int = DEFAULT_MAX_POWERLAW_SAMPLE,
) -> DatabaseTopologyStats:
    import numpy as np

    from plurel.topology_metrics import compute_edge_metrics

    records: list[EdgeTopologyRecord] = []
    total_child_rows = 0

    for child_table_name in sorted(db.table_dict):
        child_table = db.table_dict[child_table_name]
        child_df = child_table.df
        total_child_rows += len(child_df)

        for fkey_col, parent_table_name in sorted(child_table.fkey_col_to_pkey_table.items()):
            parent_table = db.table_dict[parent_table_name]
            parent_idx, null_mask = _prepare_parent_index(child_df[fkey_col])
            timestamps = _extract_timestamps(child_df, child_table.time_col)
            metrics = compute_edge_metrics(
                parent_idx=parent_idx,
                num_parents=len(parent_table.df),
                null_mask=null_mask,
                timestamps=timestamps,
                max_powerlaw_sample=max_powerlaw_sample,
            )
            records.append(
                EdgeTopologyRecord(
                    db_name=db_name,
                    child_table=child_table_name,
                    fkey_col=fkey_col,
                    parent_table=parent_table_name,
                    num_children=len(child_df),
                    num_parents=len(parent_table.df),
                    num_non_null_edges=int(np.count_nonzero(~null_mask)),
                    metrics=metrics,
                    is_self_loop=(child_table_name == parent_table_name),
                )
            )

    return DatabaseTopologyStats(
        db_name=db_name,
        num_tables=len(db.table_dict),
        num_edges=len(records),
        total_child_rows=total_child_rows,
        total_non_null_edges=sum(record.num_non_null_edges for record in records),
        edges=records,
    )


def _prepare_parent_index(series: Any) -> tuple[Any, Any]:
    import numpy as np

    null_mask = series.isna().to_numpy(dtype=bool)
    parent_idx = series.fillna(0).to_numpy(dtype=np.int64, copy=False)
    return parent_idx, null_mask


def _extract_timestamps(df: Any, time_col: str | None) -> Any:
    if time_col is None:
        return None
    return df[time_col].to_numpy()


def parse_db_names(raw_db_names: list[str]) -> list[str]:
    db_names: list[str] = []
    for raw_name in raw_db_names:
        db_names.extend(name.strip() for name in raw_name.split(",") if name.strip())
    if not db_names:
        raise ValueError("At least one db name is required.")
    return db_names


def main(
    data_root: Path,
    db_names: list[str],
    output_dir: Path,
    max_powerlaw_sample: int = DEFAULT_MAX_POWERLAW_SAMPLE,
    skip_existing: bool = False,
) -> None:
    data_root = data_root.expanduser()
    output_dir = output_dir.expanduser()
    db_stats: list[DatabaseTopologyStats] = []
    output_paths: dict[str, Path] = {}
    failures: dict[str, str] = {}

    for db_name in db_names:
        stats_path = output_dir / f"edge_topology_stats.{db_name}.json"
        if skip_existing and stats_path.exists():
            stats = load_database_stats(stats_path)
            if stats is not None:
                print(f"[skip] {db_name}: reusing {stats_path}", flush=True)
                db_stats.append(stats)
                output_paths[db_name] = stats_path
                write_summary(
                    db_stats=db_stats,
                    output_paths=output_paths,
                    data_root=data_root,
                    output_dir=output_dir,
                    failures=failures,
                )
                continue

        print(f"[run]  {db_name}: loading...", flush=True)
        try:
            db = load_relbench_database(db_name=db_name, data_root=data_root)
            stats = measure_database(
                db_name=db_name, db=db, max_powerlaw_sample=max_powerlaw_sample
            )
        except Exception as exc:
            failures[db_name] = f"{type(exc).__name__}: {exc}"
            print(f"[fail] {db_name}: {failures[db_name]}", flush=True)
            traceback.print_exc()
            write_summary(
                db_stats=db_stats,
                output_paths=output_paths,
                data_root=data_root,
                output_dir=output_dir,
                failures=failures,
            )
            continue

        output_path = write_database_stats(stats=stats, output_dir=output_dir)
        db_stats.append(stats)
        output_paths[db_name] = output_path
        print(f"[done] {db_name}: edges={stats.num_edges} -> {output_path}", flush=True)
        write_summary(
            db_stats=db_stats,
            output_paths=output_paths,
            data_root=data_root,
            output_dir=output_dir,
            failures=failures,
        )

    if failures:
        print(f"[warn] {len(failures)} dbs failed: {sorted(failures)}", flush=True)


_summarize_metrics = summarize_metrics
_load_database_stats = load_database_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure RelBench edge topology statistics.")
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Root directory containing per-database RelBench cache directories.",
    )
    parser.add_argument(
        "--db_names",
        nargs="+",
        required=True,
        help="Database names to measure, e.g. rel-f1 rel-stack or rel-f1,rel-stack.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for per-database JSON files and summary.json.",
    )
    parser.add_argument(
        "--max_powerlaw_sample",
        type=int,
        default=DEFAULT_MAX_POWERLAW_SAMPLE,
        help=(
            "Cap on the non-zero fanout sample used for the power-law fit per edge. "
            "Set to 0 to disable subsampling. Default 200000 bounds powerlaw.Fit "
            "xmin-scan runtime while keeping gamma MLE std error ~0.005."
        ),
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help=(
            "Skip databases whose edge_topology_stats.{db}.json already exists "
            "in --output_dir. Useful to resume after a crash or kill."
        ),
    )
    args = parser.parse_args()

    main(
        data_root=args.data_root,
        db_names=parse_db_names(args.db_names),
        output_dir=args.output_dir,
        max_powerlaw_sample=args.max_powerlaw_sample,
        skip_existing=args.skip_existing,
    )
