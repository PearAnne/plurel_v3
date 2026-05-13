from __future__ import annotations

import argparse
from pathlib import Path

from measure_edge_topology import load_relbench_database, measure_database, parse_db_names

from plurel.topology_measure import write_database_stats, write_summary


def discover_db_names(data_root: Path) -> list[str]:
    db_names = [
        path.name
        for path in sorted(data_root.expanduser().iterdir())
        if path.is_dir() and path.name.startswith("rel-synthetic-")
    ]
    if not db_names:
        raise ValueError(f"No rel-synthetic-* database directories found under {data_root}")
    return db_names


def main(data_root: Path, db_names: list[str] | None, output_dir: Path) -> None:
    data_root = data_root.expanduser()
    output_dir = output_dir.expanduser()
    if db_names is None:
        db_names = discover_db_names(data_root)

    db_stats = []
    output_paths = {}
    for db_name in db_names:
        db = load_relbench_database(db_name=db_name, data_root=data_root)
        stats = measure_database(db_name=db_name, db=db)
        output_paths[db_name] = write_database_stats(stats=stats, output_dir=output_dir)
        db_stats.append(stats)

    write_summary(
        db_stats=db_stats,
        output_paths=output_paths,
        data_root=data_root,
        output_dir=output_dir,
        failures={},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure synthetic DB edge topology statistics.")
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Directory containing rel-synthetic-* cache directories.",
    )
    parser.add_argument(
        "--db_names",
        nargs="+",
        default=None,
        help="Optional synthetic DB names. Defaults to all rel-synthetic-* under data_root.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for per-database JSON files and summary.json.",
    )
    args = parser.parse_args()

    main(
        data_root=args.data_root,
        db_names=None if args.db_names is None else parse_db_names(args.db_names),
        output_dir=args.output_dir,
    )
