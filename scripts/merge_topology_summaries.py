from __future__ import annotations

import argparse
import json
from pathlib import Path

from plurel.topology_measure import load_database_stats, write_summary
from plurel.topology_stats_paths import COMBINED_STATS_DIR, CTU_STATS_DIR, RELBENCH_STATS_DIR


def merge_topology_stats_dirs(
    input_dirs: list[Path],
    output_dir: Path,
) -> Path:
    output_dir = output_dir.expanduser()
    db_stats = []
    output_paths: dict[str, Path] = {}
    failures: dict[str, str] = {}
    data_roots: dict[str, str] = {}

    for input_dir in input_dirs:
        input_dir = input_dir.expanduser()
        summary_path = input_dir / "summary.json"
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            data_roots[input_dir.name] = str(payload.get("data_root", input_dir))
            failures.update(
                {
                    f"{input_dir.name}/{db_name}": message
                    for db_name, message in (payload.get("failures") or {}).items()
                }
            )

        for stats_path in sorted(input_dir.glob("edge_topology_stats.*.json")):
            stats = load_database_stats(stats_path)
            if stats is None:
                continue
            if stats.db_name in output_paths:
                raise ValueError(f"Duplicate database name while merging: {stats.db_name}")
            db_stats.append(stats)
            output_paths[stats.db_name] = stats_path

    if not db_stats:
        raise ValueError("No edge_topology_stats.*.json files found in the input directories.")

    summary_path = write_summary(
        db_stats=db_stats,
        output_paths=output_paths,
        data_root=json.dumps(data_roots, sort_keys=True),
        output_dir=output_dir,
        failures=failures,
    )
    return summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge per-benchmark topology stats directories into one summary.json."
    )
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        type=Path,
        default=[RELBENCH_STATS_DIR, CTU_STATS_DIR],
        help="Benchmark stats directories containing edge_topology_stats.*.json files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=COMBINED_STATS_DIR,
        help="Directory for the merged summary.json.",
    )
    args = parser.parse_args()

    summary_path = merge_topology_stats_dirs(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
    )
    print(f"wrote {summary_path}")
