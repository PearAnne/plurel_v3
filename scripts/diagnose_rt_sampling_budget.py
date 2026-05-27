from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

MAX_F2P_NBRS = 5
REQUIRED_PRE_FILES = ("table_info.json", "nodes.rkyv", "offsets.rkyv", "p2f_adj.rkyv")
SYNTHETIC_DB_RE = re.compile(r"^rel-synthetic-(?P<cohort>.+)-(?P<seed>\d+)$")


@dataclass(frozen=True)
class DiagnosticStatus:
    status: str
    reason: str
    configured_value: int | None = None


@dataclass(frozen=True)
class TableSlotPressure:
    db_name: str
    child_table: str
    fk_slots: int
    status: str
    fkey_cols: list[str]


@dataclass(frozen=True)
class DatabaseSlotPressure:
    db_name: str
    cohort: str
    num_tables_with_fk: int
    max_fk_slots: int
    tables_at_limit: int
    tables_over_limit: int
    table_pressures: list[TableSlotPressure]


@dataclass(frozen=True)
class BfsWidthPressure:
    db_name: str
    child_table: str
    fkey_col: str
    parent_table: str
    fanout_max: float
    max_bfs_width: int
    status: str


@dataclass(frozen=True)
class PreprocessedFileCoverage:
    db_name: str
    pre_dir: str
    present_files: list[str]
    missing_files: list[str]


@dataclass(frozen=True)
class CohortDiagnostic:
    cohort: str
    db_count: int
    max_fk_slots: int
    tables_at_f2p_limit: int
    tables_over_f2p_limit: int
    dbs_over_f2p_limit: list[str]
    dbs_at_f2p_limit: list[str]
    bfs_width_edges_over_budget: int
    bfs_width_status: DiagnosticStatus
    ctx_len_status: DiagnosticStatus


def load_manifest(manifest_path: Path | None) -> dict[str, Any]:
    if manifest_path is None:
        return {}
    return _load_json(manifest_path.expanduser())


def load_database_stats_paths(stats_paths: list[Path], stats_dirs: list[Path]) -> list[Path]:
    paths = [path.expanduser() for path in stats_paths]
    for stats_dir in stats_dirs:
        paths.extend(sorted(stats_dir.expanduser().glob("**/edge_topology_stats.*.json")))
    unique_paths = sorted({path.resolve() for path in paths})
    if not unique_paths:
        raise ValueError("No topology stats files were found.")
    return unique_paths


def build_diagnostic(
    stats_paths: list[Path],
    manifest: dict[str, Any] | None = None,
    pre_root: Path | None = None,
    max_bfs_width: int | None = None,
    ctx_len: int | None = None,
    f2p_limit: int = MAX_F2P_NBRS,
) -> dict[str, Any]:
    db_records = [_load_database_stats(path) for path in stats_paths]
    manifest_cohorts = _manifest_cohort_lookup(manifest or {})
    db_pressures = [
        _diagnose_db_slot_pressure(
            db_record=record,
            cohort=_cohort_for_db(record["db_name"], manifest_cohorts),
            f2p_limit=f2p_limit,
        )
        for record in db_records
    ]
    bfs_pressures = _diagnose_bfs_width_pressure(
        db_records=db_records,
        max_bfs_width=max_bfs_width,
    )
    pre_coverage = (
        [_diagnose_preprocessed_files(record["db_name"], pre_root) for record in db_records]
        if pre_root is not None
        else []
    )
    cohort_diagnostics = _summarize_cohorts(
        db_pressures=db_pressures,
        bfs_pressures=bfs_pressures,
        max_bfs_width=max_bfs_width,
        ctx_len=ctx_len,
    )
    return {
        "schema_version": 1,
        "limits": {
            "max_f2p_nbrs": f2p_limit,
            "max_bfs_width": max_bfs_width,
            "ctx_len": ctx_len,
        },
        "cohorts": [asdict(cohort) for cohort in cohort_diagnostics],
        "databases": [asdict(pressure) for pressure in db_pressures],
        "bfs_width_pressure": [asdict(pressure) for pressure in bfs_pressures],
        "preprocessed_file_coverage": [asdict(coverage) for coverage in pre_coverage],
    }


def write_diagnostic(diagnostic: dict[str, Any], output_path: Path | None) -> None:
    text = json.dumps(diagnostic, indent=2, sort_keys=True)
    if output_path is None:
        LOGGER.info("%s", text)
        return
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{text}\n", encoding="utf-8")
    LOGGER.info("wrote diagnostic to %s", output_path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc


def _load_database_stats(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    required_keys = {"db_name", "edges"}
    missing_keys = required_keys - payload.keys()
    if missing_keys:
        raise ValueError(f"{path} is not a database topology stats file; missing {missing_keys}")
    if not isinstance(payload["edges"], list):
        raise ValueError(f"{path} has invalid edges; expected a list")
    return payload


def _manifest_cohort_lookup(manifest: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row in manifest.get("cohorts", []):
        cohort = row.get("cohort")
        seeds = row.get("seeds", [])
        if not isinstance(cohort, str) or not isinstance(seeds, list):
            continue
        for seed in seeds:
            lookup[f"rel-synthetic-{cohort}-{seed}"] = cohort
    return lookup


def _cohort_for_db(db_name: str, manifest_cohorts: dict[str, str]) -> str:
    if db_name in manifest_cohorts:
        return manifest_cohorts[db_name]
    match = SYNTHETIC_DB_RE.match(db_name)
    if match is not None:
        return match.group("cohort")
    return "unassigned"


def _diagnose_db_slot_pressure(
    db_record: dict[str, Any],
    cohort: str,
    f2p_limit: int,
) -> DatabaseSlotPressure:
    table_to_fkeys: dict[str, set[str]] = {}
    for edge in db_record["edges"]:
        child_table = _required_str(edge, "child_table")
        fkey_col = _required_str(edge, "fkey_col")
        table_to_fkeys.setdefault(child_table, set()).add(fkey_col)

    table_pressures: list[TableSlotPressure] = []
    for child_table, fkey_cols in sorted(table_to_fkeys.items()):
        fk_slots = len(fkey_cols)
        table_pressures.append(
            TableSlotPressure(
                db_name=db_record["db_name"],
                child_table=child_table,
                fk_slots=fk_slots,
                status=_slot_status(fk_slots=fk_slots, f2p_limit=f2p_limit),
                fkey_cols=sorted(fkey_cols),
            )
        )

    return DatabaseSlotPressure(
        db_name=db_record["db_name"],
        cohort=cohort,
        num_tables_with_fk=len(table_pressures),
        max_fk_slots=max((pressure.fk_slots for pressure in table_pressures), default=0),
        tables_at_limit=sum(pressure.fk_slots == f2p_limit for pressure in table_pressures),
        tables_over_limit=sum(pressure.fk_slots > f2p_limit for pressure in table_pressures),
        table_pressures=table_pressures,
    )


def _slot_status(fk_slots: int, f2p_limit: int) -> str:
    if fk_slots > f2p_limit:
        return "over_limit"
    if fk_slots == f2p_limit:
        return "at_limit"
    return "ok"


def _diagnose_bfs_width_pressure(
    db_records: list[dict[str, Any]],
    max_bfs_width: int | None,
) -> list[BfsWidthPressure]:
    if max_bfs_width is None:
        return []
    pressures: list[BfsWidthPressure] = []
    for record in db_records:
        for edge in record["edges"]:
            metrics = edge.get("metrics", {})
            fanout_max = _optional_float(metrics.get("fanout_max"))
            if fanout_max is None or fanout_max <= max_bfs_width:
                continue
            pressures.append(
                BfsWidthPressure(
                    db_name=record["db_name"],
                    child_table=_required_str(edge, "child_table"),
                    fkey_col=_required_str(edge, "fkey_col"),
                    parent_table=_required_str(edge, "parent_table"),
                    fanout_max=fanout_max,
                    max_bfs_width=max_bfs_width,
                    status="static_upper_bound_over_budget",
                )
            )
    return pressures


def _diagnose_preprocessed_files(db_name: str, pre_root: Path) -> PreprocessedFileCoverage:
    pre_dir = pre_root.expanduser() / db_name
    present_files = [name for name in REQUIRED_PRE_FILES if (pre_dir / name).exists()]
    missing_files = [name for name in REQUIRED_PRE_FILES if name not in present_files]
    return PreprocessedFileCoverage(
        db_name=db_name,
        pre_dir=str(pre_dir),
        present_files=present_files,
        missing_files=missing_files,
    )


def _summarize_cohorts(
    db_pressures: list[DatabaseSlotPressure],
    bfs_pressures: list[BfsWidthPressure],
    max_bfs_width: int | None,
    ctx_len: int | None,
) -> list[CohortDiagnostic]:
    pressures_by_cohort: dict[str, list[DatabaseSlotPressure]] = {}
    for pressure in db_pressures:
        pressures_by_cohort.setdefault(pressure.cohort, []).append(pressure)

    bfs_count_by_db: dict[str, int] = {}
    for pressure in bfs_pressures:
        bfs_count_by_db[pressure.db_name] = bfs_count_by_db.get(pressure.db_name, 0) + 1

    summaries: list[CohortDiagnostic] = []
    for cohort, pressures in sorted(pressures_by_cohort.items()):
        summaries.append(
            CohortDiagnostic(
                cohort=cohort,
                db_count=len(pressures),
                max_fk_slots=max((pressure.max_fk_slots for pressure in pressures), default=0),
                tables_at_f2p_limit=sum(pressure.tables_at_limit for pressure in pressures),
                tables_over_f2p_limit=sum(pressure.tables_over_limit for pressure in pressures),
                dbs_over_f2p_limit=[
                    pressure.db_name for pressure in pressures if pressure.tables_over_limit > 0
                ],
                dbs_at_f2p_limit=[
                    pressure.db_name for pressure in pressures if pressure.tables_at_limit > 0
                ],
                bfs_width_edges_over_budget=sum(
                    bfs_count_by_db.get(pressure.db_name, 0) for pressure in pressures
                ),
                bfs_width_status=_bfs_width_status(max_bfs_width=max_bfs_width),
                ctx_len_status=DiagnosticStatus(
                    status="unavailable",
                    reason=(
                        "Existing topology stats and preprocessed files do not record sampled "
                        "sequence occupancy or ctx_len truncation counts; sampler instrumentation "
                        "or a sampled batch trace is needed."
                    ),
                    configured_value=ctx_len,
                ),
            )
        )
    return summaries


def _bfs_width_status(max_bfs_width: int | None) -> DiagnosticStatus:
    if max_bfs_width is None:
        return DiagnosticStatus(
            status="unavailable",
            reason="No max_bfs_width was provided, so static fanout pressure was not evaluated.",
        )
    return DiagnosticStatus(
        status="static_upper_bound",
        reason=(
            "Compared edge-level fanout_max to max_bfs_width. This can flag possible p2f frontier "
            "pressure, but it is not a runtime truncation rate because timestamps and sampled BFS "
            "frontiers are not observed."
        ),
        configured_value=max_bfs_width,
    )


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Expected string field '{key}', got {value!r}")
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose RT sampling budget pressure from existing stats and metadata."
    )
    parser.add_argument(
        "--stats_path",
        type=Path,
        action="append",
        default=[],
        help="Database topology stats JSON. Can be passed multiple times.",
    )
    parser.add_argument(
        "--stats_dir",
        type=Path,
        action="append",
        default=[],
        help="Directory searched recursively for edge_topology_stats.*.json files.",
    )
    parser.add_argument("--manifest_path", type=Path, default=None)
    parser.add_argument("--pre_root", type=Path, default=None)
    parser.add_argument("--max_bfs_width", type=int, default=None)
    parser.add_argument("--ctx_len", type=int, default=None)
    parser.add_argument("--output_path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    stats_paths = load_database_stats_paths(
        stats_paths=args.stats_path,
        stats_dirs=args.stats_dir,
    )
    diagnostic = build_diagnostic(
        stats_paths=stats_paths,
        manifest=load_manifest(args.manifest_path),
        pre_root=args.pre_root,
        max_bfs_width=args.max_bfs_width,
        ctx_len=args.ctx_len,
    )
    write_diagnostic(diagnostic=diagnostic, output_path=args.output_path)


if __name__ == "__main__":
    main()
