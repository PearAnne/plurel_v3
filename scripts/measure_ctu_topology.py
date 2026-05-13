from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from measure_edge_topology import parse_db_names

from plurel.topology_adapters import discover_ctu_database_names, load_ctu_database
from plurel.topology_measure import (
    load_database_stats,
    measure_topology_database,
    write_database_stats,
    write_summary,
)
from plurel.topology_metrics import DEFAULT_MAX_POWERLAW_SAMPLE
from plurel.topology_stats_paths import CTU_STATS_DIR


def main(
    data_root: Path,
    db_names: list[str] | None,
    output_dir: Path,
    max_powerlaw_sample: int = DEFAULT_MAX_POWERLAW_SAMPLE,
    skip_existing: bool = False,
) -> None:
    data_root = data_root.expanduser()
    output_dir = output_dir.expanduser()
    if db_names is None:
        db_names = discover_ctu_database_names(data_root)

    db_stats = []
    output_paths: dict[str, Path] = {}
    failures: dict[str, str] = {}

    for db_name in db_names:
        stats_path = output_dir / f"edge_topology_stats.ctu-{db_name}.json"
        if skip_existing and stats_path.exists():
            stats = load_database_stats(stats_path)
            if stats is not None:
                print(f"[skip] ctu-{db_name}: reusing {stats_path}", flush=True)
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

        print(f"[run]  ctu-{db_name}: loading...", flush=True)
        try:
            database = load_ctu_database(ctu_root=data_root, db_name=db_name)
            stats = measure_topology_database(
                database=database,
                max_powerlaw_sample=max_powerlaw_sample,
            )
        except Exception as exc:
            failures[db_name] = f"{type(exc).__name__}: {exc}"
            print(f"[fail] ctu-{db_name}: {failures[db_name]}", flush=True)
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
            f"[done] {stats.db_name}: edges={stats.num_edges} -> {output_path}",
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
    parser = argparse.ArgumentParser(description="Measure CTU edge topology statistics.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/local/lzd/plurel_runtime/relbench/ctu"),
        help="Root directory containing CTU database folders.",
    )
    parser.add_argument(
        "--db_names",
        nargs="+",
        default=None,
        help="Optional CTU database folder names. Defaults to all databases under data_root.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=CTU_STATS_DIR,
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
        help="Skip databases whose edge_topology_stats.ctu-*.json already exists.",
    )
    args = parser.parse_args()

    main(
        data_root=args.data_root,
        db_names=None if args.db_names is None else parse_db_names(args.db_names),
        output_dir=args.output_dir,
        max_powerlaw_sample=args.max_powerlaw_sample,
        skip_existing=args.skip_existing,
    )
