from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rfm.rdb.types import ForeignKeySpec, RelationalDataset


@dataclass(frozen=True)
class InvariantRange:
    min_value: float | None
    max_value: float | None
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None
    source: str = "combined"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MechanismInvariantBounds:
    source: str
    num_databases: int
    num_edges: int
    metrics: dict[str, InvariantRange]
    cardinality_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "num_databases": self.num_databases,
            "num_edges": self.num_edges,
            "metrics": {key: value.to_dict() for key, value in self.metrics.items()},
            "cardinality_counts": self.cardinality_counts,
        }


@dataclass(frozen=True)
class CalibrationCheck:
    metric: str
    synthetic_value: float
    bound_min: float | None
    bound_max: float | None
    passed: bool
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationGateReport:
    passed: bool
    checks: tuple[CalibrationCheck, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
            "warnings": list(self.warnings),
        }


DEFAULT_METRICS: tuple[str, ...] = (
    "fanout_gini",
    "null_rate",
    "true_null_rate",
    "fanout_ks_to_poisson",
    "fanout_max",
    "fanout_p50",
)


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_bounds_from_summary(summary: dict[str, Any], source: str) -> MechanismInvariantBounds:
    metric_summary = summary.get("metric_summary", {})
    cardinality = metric_summary.get("cardinality_kind", {}).get("value_counts", {})
    metrics: dict[str, InvariantRange] = {}
    for metric_name in DEFAULT_METRICS:
        stats = metric_summary.get(metric_name, {})
        if not stats:
            continue
        metrics[metric_name] = InvariantRange(
            min_value=_optional_float(stats.get("p10")),
            max_value=_optional_float(stats.get("p90")),
            p10=_optional_float(stats.get("p10")),
            p50=_optional_float(stats.get("p50")),
            p90=_optional_float(stats.get("p90")),
            source=source,
        )
    num_edges = int(metric_summary.get("fanout_gini", {}).get("count", 0))
    num_databases = len(summary.get("db_names", summary.get("dbs", {})))
    return MechanismInvariantBounds(
        source=source,
        num_databases=num_databases,
        num_edges=num_edges,
        metrics=metrics,
        cardinality_counts={str(key): int(value) for key, value in cardinality.items()},
    )


def load_bounds_bundle(path: Path) -> dict[str, MechanismInvariantBounds]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    bundle: dict[str, MechanismInvariantBounds] = {}
    for source, item in payload.get("sources", {}).items():
        metrics = {
            name: InvariantRange(**metric) for name, metric in item.get("metrics", {}).items()
        }
        bundle[source] = MechanismInvariantBounds(
            source=source,
            num_databases=int(item.get("num_databases", 0)),
            num_edges=int(item.get("num_edges", 0)),
            metrics=metrics,
            cardinality_counts={
                str(key): int(value) for key, value in item.get("cardinality_counts", {}).items()
            },
        )
    return bundle


def save_bounds_bundle(path: Path, bounds: Mapping[str, MechanismInvariantBounds]) -> None:
    payload = {
        "description": "Mechanism-invariant sanity bounds derived from real RDB topology statistics.",
        "sources": {source: bound.to_dict() for source, bound in bounds.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compute_synthetic_aggregate_metrics(databases: Sequence[RelationalDataset]) -> dict[str, float]:
    if len(databases) == 0:
        raise ValueError("databases must be non-empty")

    gini_values: list[float] = []
    null_rates: list[float] = []
    poisson_ks: list[float] = []
    fanout_max: list[float] = []
    fanout_p50: list[float] = []

    for database in databases:
        for fk in database.foreign_keys:
            gini_values.append(_fanout_gini(database, fk))
            values = database.column_values(fk.child_table, fk.child_column)
            null_rates.append(float(np.mean(~np.isfinite(values))))
            fanout = np.bincount(
                values[np.isfinite(values)].astype(np.int64),
                minlength=database.table_specs[fk.parent_table].row_count,
            )
            positive = fanout[fanout > 0]
            if len(positive) > 0:
                fanout_max.append(float(np.max(positive)))
                fanout_p50.append(float(np.median(positive)))
            poisson_ks.append(_approx_poisson_ks(positive if len(positive) > 0 else fanout))

    return {
        "fanout_gini": float(np.median(gini_values)) if gini_values else 0.0,
        "null_rate": float(np.mean(null_rates)) if null_rates else 0.0,
        "true_null_rate": float(np.mean(null_rates)) if null_rates else 0.0,
        "fanout_ks_to_poisson": float(np.median(poisson_ks)) if poisson_ks else 0.0,
        "fanout_max": float(np.median(fanout_max)) if fanout_max else 0.0,
        "fanout_p50": float(np.median(fanout_p50)) if fanout_p50 else 0.0,
        "mean_tables": float(np.mean([len(db.table_specs) for db in databases])),
        "mean_foreign_keys": float(np.mean([len(db.foreign_keys) for db in databases])),
    }


def evaluate_calibration_gate(
    databases: Sequence[RelationalDataset],
    bounds: MechanismInvariantBounds,
    *,
    allow_warnings: bool = True,
) -> CalibrationGateReport:
    synthetic = compute_synthetic_aggregate_metrics(databases)
    checks: list[CalibrationCheck] = []
    warnings: list[str] = []

    for metric_name, metric_bounds in bounds.metrics.items():
        if metric_name not in synthetic:
            continue
        value = float(synthetic[metric_name])
        lower = metric_bounds.min_value
        upper = metric_bounds.max_value
        passed = True
        note = ""
        if lower is not None and value < lower:
            passed = False
            note = f"{value:.4f} < p10 bound {lower:.4f}"
        if upper is not None and value > upper:
            passed = False
            note = f"{value:.4f} > p90 bound {upper:.4f}"
        checks.append(
            CalibrationCheck(
                metric=metric_name,
                synthetic_value=value,
                bound_min=lower,
                bound_max=upper,
                passed=passed,
                note=note,
            )
        )
        if not passed and allow_warnings:
            warnings.append(f"calibration warning for {metric_name}: {note}")

    coverage_checks = _coverage_checks(databases)
    checks.extend(coverage_checks)
    passed = all(check.passed for check in checks if check.metric.startswith("coverage_"))
    if not passed and allow_warnings:
        warnings.extend(check.note for check in checks if not check.passed and check.note)
    return CalibrationGateReport(passed=passed, checks=tuple(checks), warnings=tuple(warnings))


def _coverage_checks(databases: Sequence[RelationalDataset]) -> list[CalibrationCheck]:
    ginis = []
    has_low = False
    has_high = False
    for database in databases:
        for fk in database.foreign_keys:
            gini = _fanout_gini(database, fk)
            ginis.append(gini)
            if gini < 0.35:
                has_low = True
            if gini > 0.55:
                has_high = True
    return [
        CalibrationCheck(
            metric="coverage_low_gini_present",
            synthetic_value=float(has_low),
            bound_min=1.0,
            bound_max=1.0,
            passed=has_low,
            note="" if has_low else "no low-gini edge observed",
        ),
        CalibrationCheck(
            metric="coverage_high_gini_present",
            synthetic_value=float(has_high),
            bound_min=1.0,
            bound_max=1.0,
            passed=has_high,
            note="" if has_high else "no high-gini edge observed",
        ),
    ]


def _approx_poisson_ks(fanout: NDArray[np.int64]) -> float:
    if fanout.size == 0:
        return 0.0
    empirical = np.sort(fanout.astype(np.float64) / max(float(np.sum(fanout)), 1.0))
    mean = float(np.mean(fanout))
    if mean <= 1e-12:
        return 0.0
    poisson_cdf = np.array(
        [_poisson_cdf(value, mean) for value in range(fanout.size)], dtype=np.float64
    )
    poisson_cdf = poisson_cdf / max(float(poisson_cdf[-1]), 1e-12)
    grid = np.linspace(0.0, 1.0, fanout.size)
    return float(np.max(np.abs(empirical - poisson_cdf)))


def _poisson_cdf(k: int, mean: float) -> float:
    total = 0.0
    for idx in range(k + 1):
        total += float(np.exp(-mean) * (mean**idx) / max(float(math.factorial(idx)), 1.0))
    return total


def _fanout_gini(database: RelationalDataset, fk: ForeignKeySpec) -> float:
    values = database.column_values(fk.child_table, fk.child_column)
    parent_rows = values[np.isfinite(values)].astype(np.int64)
    if len(parent_rows) == 0:
        return 0.0
    counts = np.bincount(
        parent_rows, minlength=database.table_specs[fk.parent_table].row_count
    ).astype(np.float64)
    if float(counts.sum()) <= 0.0:
        return 0.0
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    cumulative = np.cumsum(sorted_counts)
    return float((n + 1 - 2.0 * np.sum(cumulative) / cumulative[-1]) / n)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
