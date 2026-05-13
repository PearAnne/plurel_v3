from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from measure_edge_topology import parse_db_names

from plurel.topology_adapters import (
    DBINFER_NONOVERLAPPING_DATASETS,
    load_dbinfer_database,
    normalize_dbinfer_dataset_name,
)
from plurel.topology_measure import (
    load_database_stats,
    measure_topology_database,
    write_database_stats,
    write_summary,
)
from plurel.topology_metrics import DEFAULT_MAX_POWERLAW_SAMPLE
from plurel.topology_stats_paths import DBINFER_STATS_DIR


def main(
    data_root: Path,
    dataset_names: list[str],
    output_dir: Path,
    max_powerlaw_sample: int = DEFAULT_MAX_POWERLAW_SAMPLE,
    skip_existing: bool = False,
) -> None:
    data_root = data_root.expanduser()
    output_dir = output_dir.expanduser()

    db_stats = []
    output_paths: dict[str, Path] = {}
    failures: dict[str, str] = {}

    for dataset_name in dataset_names:
        db_name = f"dbinfer-{dataset_name}"
        stats_path = output_dir / f"edge_topology_stats.{db_name}.json"
        if skip_existing and stats_path.exists():
            stats = load_database_stats(stats_path)
            if stats is not None:
                print(f"[skip] {db_name}: reusing {stats_path}", flush=True)
                db_stats.append(stats)
                output_paths[stats.db_name] = stats_path
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
            database = load_dbinfer_database(data_root=data_root, dataset_name=dataset_name)
            if not database.foreign_keys:
                failures[dataset_name] = "ValueError: no FK edges discovered"
                print(f"[fail] {db_name}: {failures[dataset_name]}", flush=True)
                write_summary(
                    db_stats=db_stats,
                    output_paths=output_paths,
                    data_root=data_root,
                    output_dir=output_dir,
                    failures=failures,
                )
                continue
            stats = measure_topology_database(
                database=database,
                max_powerlaw_sample=max_powerlaw_sample,
            )
        except Exception as exc:
            failures[dataset_name] = f"{type(exc).__name__}: {exc}"
            print(f"[fail] {db_name}: {failures[dataset_name]}", flush=True)
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
        output_paths[stats.db_name] = output_path
        print(
            f"[done] {db_name}: edges={stats.num_edges} -> {output_path}",
            flush=True,
        )
        write_summary(
            db_stats=db_stats,
            output_paths=output_paths,
            data_root=data_root,
            output_dir=output_dir,
            failures=failures,
        )

    if failures:
        print(f"[warn] {len(failures)} dbs failed: {sorted(failures)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure 4DBInfer edge topology statistics.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/local/lzd/plurel_runtime/relbench"),
        help="Root directory containing dbinfer-* dataset folders.",
    )
    parser.add_argument(
        "--dataset_names",
        nargs="+",
        default=list(DBINFER_NONOVERLAPPING_DATASETS),
        help="4DBInfer dataset names without the dbinfer- prefix.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DBINFER_STATS_DIR,
        help="Directory for per-database JSON files and summary.json.",
    )
    parser.add_argument(
        "--max_powerlaw_sample",
        type=int,
        default=DEFAULT_MAX_POWERLAW_SAMPLE,
        help="Cap on the non-zero fanout sample used for the power-law fit per edge.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip datasets whose edge_topology_stats.dbinfer-*.json already exists.",
    )
    args = parser.parse_args()

    dataset_names = [
        normalize_dbinfer_dataset_name(name) for name in parse_db_names(args.dataset_names)
    ]
    main(
        data_root=args.data_root,
        dataset_names=dataset_names,
        output_dir=args.output_dir,
        max_powerlaw_sample=args.max_powerlaw_sample,
        skip_existing=args.skip_existing,
    )
