from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plurel.topology_metrics import DEFAULT_MAX_POWERLAW_SAMPLE


@dataclass(frozen=True)
class EdgeTopologyRecord:
    db_name: str
    child_table: str
    fkey_col: str
    parent_table: str
    num_children: int
    num_parents: int
    num_non_null_edges: int
    metrics: dict[str, float | int | str | bool | None]
    num_true_null_edges: int = 0
    num_unmatched_fk_edges: int = 0
    is_self_loop: bool = False


@dataclass(frozen=True)
class DatabaseTopologyStats:
    db_name: str
    num_tables: int
    num_edges: int
    total_child_rows: int
    total_non_null_edges: int
    edges: list[EdgeTopologyRecord]


@dataclass(frozen=True)
class ForeignKeyEdge:
    child_table: str
    fkey_col: str
    parent_table: str
    parent_pkey_col: str


@dataclass(frozen=True)
class TopologyTable:
    df: pd.DataFrame
    time_col: str | None = None


@dataclass(frozen=True)
class TopologyDatabase:
    db_name: str
    tables: dict[str, TopologyTable]
    foreign_keys: list[ForeignKeyEdge]


def fk_values_to_parent_row_indices(
    fk: pd.Series,
    parent_df: pd.DataFrame,
    parent_pkey_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    parent_idx, null_mask, _, _ = fk_values_to_parent_row_indices_with_quality(
        fk=fk,
        parent_df=parent_df,
        parent_pkey_col=parent_pkey_col,
    )
    return parent_idx, null_mask


def fk_values_to_parent_row_indices_with_quality(
    fk: pd.Series,
    parent_df: pd.DataFrame,
    parent_pkey_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map FK values to parent row indices via a left join (avoids ``Series.map`` edge cases)."""
    row_positions = np.arange(len(parent_df), dtype=np.int64)
    mapping = pd.DataFrame(
        {"_k": parent_df[parent_pkey_col].to_numpy(), "_row": row_positions}
    ).drop_duplicates(subset=["_k"], keep="first")
    left = pd.DataFrame({"_k": fk.reset_index(drop=True)})
    try:
        merged = left.merge(mapping, on="_k", how="left", sort=False)
    except ValueError:
        left, mapping = _coerce_fk_merge_frames(fk=fk, parent_key=parent_df[parent_pkey_col])
        mapping["_row"] = row_positions
        mapping = mapping.drop_duplicates(subset=["_k"], keep="first")
        merged = left.merge(mapping, on="_k", how="left", sort=False)
    null_mask_fk = fk.isna().to_numpy(dtype=bool)
    row_vals = merged["_row"]
    null_mask_mapper = row_vals.isna().to_numpy(dtype=bool)
    unmatched_fk_mask = null_mask_mapper & ~null_mask_fk
    null_mask = null_mask_fk | unmatched_fk_mask
    safe_rows = pd.to_numeric(row_vals, errors="coerce").fillna(0.0)
    parent_idx = np.where(null_mask, 0, safe_rows).astype(np.int64, copy=False)
    return parent_idx, null_mask, null_mask_fk, unmatched_fk_mask


def _coerce_fk_merge_frames(
    fk: pd.Series, parent_key: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fk_key = fk.reset_index(drop=True)
    parent_key = parent_key.reset_index(drop=True)
    fk_numeric = pd.to_numeric(fk_key, errors="coerce")
    parent_numeric = pd.to_numeric(parent_key, errors="coerce")
    if fk_numeric.notna().any() and parent_numeric.notna().any():
        return pd.DataFrame({"_k": fk_numeric}), pd.DataFrame({"_k": parent_numeric})
    return (
        pd.DataFrame({"_k": fk_key.astype("string")}),
        pd.DataFrame({"_k": parent_key.astype("string")}),
    )


def add_fk_quality_metrics(
    metrics: dict[str, float | int | str | bool | None],
    true_null_mask: np.ndarray,
    unmatched_fk_mask: np.ndarray,
) -> dict[str, float | int | str | bool | None]:
    """Record FK data-quality rates separately from the fanout exclusion mask."""
    if true_null_mask.shape != unmatched_fk_mask.shape:
        raise ValueError("true_null_mask and unmatched_fk_mask must have the same shape")
    total = int(true_null_mask.size)
    enriched = dict(metrics)
    enriched["true_null_rate"] = float(true_null_mask.mean()) if total else 0.0
    enriched["unmatched_fk_rate"] = float(unmatched_fk_mask.mean()) if total else 0.0
    return enriched


def measure_topology_database(
    database: TopologyDatabase,
    max_powerlaw_sample: int = DEFAULT_MAX_POWERLAW_SAMPLE,
) -> DatabaseTopologyStats:
    from plurel.topology_metrics import compute_edge_metrics

    if not database.foreign_keys:
        total_rows = sum(len(t.df) for t in database.tables.values())
        return DatabaseTopologyStats(
            db_name=database.db_name,
            num_tables=len(database.tables),
            num_edges=0,
            total_child_rows=total_rows,
            total_non_null_edges=0,
            edges=[],
        )

    records: list[EdgeTopologyRecord] = []
    total_child_rows = 0

    for edge in database.foreign_keys:
        child = database.tables[edge.child_table]
        parent = database.tables[edge.parent_table]
        child_df = child.df
        parent_df = parent.df
        total_child_rows += len(child_df)

        (
            parent_idx,
            null_mask,
            true_null_mask,
            unmatched_fk_mask,
        ) = fk_values_to_parent_row_indices_with_quality(
            child_df[edge.fkey_col],
            parent_df,
            edge.parent_pkey_col,
        )
        timestamps = (
            child_df[child.time_col].to_numpy()
            if child.time_col is not None and child.time_col in child_df.columns
            else None
        )
        metrics = compute_edge_metrics(
            parent_idx=parent_idx,
            num_parents=len(parent_df),
            null_mask=null_mask,
            timestamps=timestamps,
            max_powerlaw_sample=max_powerlaw_sample,
        )
        metrics = add_fk_quality_metrics(
            metrics=metrics,
            true_null_mask=true_null_mask,
            unmatched_fk_mask=unmatched_fk_mask,
        )
        records.append(
            EdgeTopologyRecord(
                db_name=database.db_name,
                child_table=edge.child_table,
                fkey_col=edge.fkey_col,
                parent_table=edge.parent_table,
                num_children=len(child_df),
                num_parents=len(parent_df),
                num_non_null_edges=int(np.count_nonzero(~null_mask)),
                metrics=metrics,
                num_true_null_edges=int(np.count_nonzero(true_null_mask)),
                num_unmatched_fk_edges=int(np.count_nonzero(unmatched_fk_mask)),
                is_self_loop=(edge.child_table == edge.parent_table),
            )
        )

    return DatabaseTopologyStats(
        db_name=database.db_name,
        num_tables=len(database.tables),
        num_edges=len(records),
        total_child_rows=total_child_rows,
        total_non_null_edges=sum(record.num_non_null_edges for record in records),
        edges=records,
    )


SUMMARY_PERCENTILES: tuple[int, ...] = (10, 25, 50, 75, 90)


def write_database_stats(stats: DatabaseTopologyStats, output_dir: Path) -> Path:
    output_path = output_dir / f"edge_topology_stats.{stats.db_name}.json"
    _write_json(output_path, asdict(stats))
    return output_path


def build_summary(
    db_stats: list[DatabaseTopologyStats],
    output_paths: dict[str, Path],
    data_root: Path | str,
    output_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for stats in db_stats:
        for edge in stats.edges:
            row = {
                "db_name": stats.db_name,
                "stats_path": str(output_paths[stats.db_name]),
                "num_tables": stats.num_tables,
                "num_edges": stats.num_edges,
                "total_child_rows": stats.total_child_rows,
                "total_non_null_edges": stats.total_non_null_edges,
                **asdict(edge),
            }
            row["metrics"] = _json_safe(row["metrics"])
            rows.append(row)

    return {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "num_dbs": len(db_stats),
        "db_names": [stats.db_name for stats in db_stats],
        "rows": rows,
        "dbs": {
            stats.db_name: {
                "stats_path": str(output_paths[stats.db_name]),
                "num_tables": stats.num_tables,
                "num_edges": stats.num_edges,
                "total_child_rows": stats.total_child_rows,
                "total_non_null_edges": stats.total_non_null_edges,
            }
            for stats in db_stats
        },
        "metric_summary": summarize_metrics(db_stats),
        "metric_summary_plausible": summarize_metrics(db_stats, plausible_only=True),
    }


def summarize_metrics(
    db_stats: list[DatabaseTopologyStats],
    plausible_only: bool = False,
) -> dict[str, dict[str, float | int]]:
    values_by_metric: dict[str, list[float]] = {}
    missing_by_metric: dict[str, int] = {}
    categorical_counts: dict[str, dict[str, int]] = {}
    for stats in db_stats:
        for edge in stats.edges:
            if plausible_only and not edge.metrics.get("powerlaw_plausible"):
                continue
            for metric_name, metric_value in edge.metrics.items():
                if isinstance(metric_value, bool):
                    coerced: float | None = float(metric_value)
                elif metric_value is None:
                    coerced = None
                elif isinstance(metric_value, str):
                    bucket = categorical_counts.setdefault(metric_name, {})
                    bucket[metric_value] = bucket.get(metric_value, 0) + 1
                    continue
                else:
                    try:
                        coerced = float(metric_value)
                    except (TypeError, ValueError):
                        coerced = None
                    else:
                        if not np.isfinite(coerced):
                            coerced = None
                if coerced is None:
                    missing_by_metric[metric_name] = missing_by_metric.get(metric_name, 0) + 1
                    continue
                values_by_metric.setdefault(metric_name, []).append(coerced)

    summary: dict[str, dict[str, float | int]] = {}
    for metric_name, values in sorted(values_by_metric.items()):
        arr = np.asarray(values, dtype=float)
        record: dict[str, float | int] = {
            "count": int(arr.size),
            "n_missing": int(missing_by_metric.get(metric_name, 0)),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
        for percentile in SUMMARY_PERCENTILES:
            record[f"p{percentile:02d}"] = float(np.quantile(arr, percentile / 100.0))
        summary[metric_name] = record
    for metric_name, n_missing in missing_by_metric.items():
        if metric_name not in summary:
            summary[metric_name] = {
                "count": 0,
                "n_missing": int(n_missing),
            }
    for metric_name, buckets in categorical_counts.items():
        total = sum(buckets.values())
        summary[metric_name] = {
            "count": int(total),
            "n_missing": int(missing_by_metric.get(metric_name, 0)),
            "value_counts": dict(sorted(buckets.items())),
        }
    return summary


def load_database_stats(stats_path: Path) -> DatabaseTopologyStats | None:
    try:
        payload = json.loads(stats_path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        edges = [EdgeTopologyRecord(**edge) for edge in payload["edges"]]
        return DatabaseTopologyStats(
            db_name=payload["db_name"],
            num_tables=payload["num_tables"],
            num_edges=payload["num_edges"],
            total_child_rows=payload["total_child_rows"],
            total_non_null_edges=payload["total_non_null_edges"],
            edges=edges,
        )
    except (TypeError, KeyError):
        return None


def write_summary(
    db_stats: list[DatabaseTopologyStats],
    output_paths: dict[str, Path],
    data_root: Path | str,
    output_dir: Path,
    failures: dict[str, str] | None = None,
) -> Path:
    summary = build_summary(
        db_stats=db_stats,
        output_paths=output_paths,
        data_root=data_root,
        output_dir=output_dir,
    )
    summary["failures"] = dict(failures or {})
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    return summary_path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
