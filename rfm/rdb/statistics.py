from __future__ import annotations

import zlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace

import numpy as np
from numpy.typing import NDArray

from rfm.rdb.config import (
    MechanismHyperpriorConfig,
    RDBPriorConfig,
    RoleGrammarConfig,
    SchemaArchetypeConfig,
)
from rfm.rdb.generator import RelationalPriorGenerator
from rfm.rdb.invariants import (
    CalibrationGateReport,
    MechanismInvariantBounds,
    evaluate_calibration_gate,
)
from rfm.rdb.types import ForeignKeySpec, RelationalDataset


@dataclass(frozen=True)
class RDBStatisticsReport:
    sample_count: int
    archetype_counts: dict[str, int]
    role_counts: dict[str, int]
    attachment_counts: dict[str, int]
    existence_counts: dict[str, int]
    intent_counts: dict[str, int]
    mean_tables: float
    mean_foreign_keys: float
    mean_structural_null_rate: float
    mean_feature_missing_rate: float
    regime_hits: dict[str, float]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RDBTopologyRegimeThresholds:
    uniform_gini_max: float = 0.3
    hub_gini_min: float = 0.7
    optional_null_min: float = 0.05
    optional_null_max: float = 0.5
    optional_mi_min: float = 0.01
    multi_parent_mi_ratio_min: float = 2.0
    min_regime_hit_count: int = 1
    min_regime_hit_rate: float = 0.0


@dataclass(frozen=True)
class RDBTopologyRegimeGate:
    passed: bool
    failures: tuple[str, ...]
    regime_hits: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_databases(databases: Sequence[RelationalDataset]) -> RDBStatisticsReport:
    if len(databases) == 0:
        raise ValueError("databases must be non-empty")

    role_counts: Counter[str] = Counter()
    archetype_counts: Counter[str] = Counter()
    attachment_counts: Counter[str] = Counter()
    existence_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    structural_null_rates: list[float] = []
    feature_missing_rates: list[float] = []
    regime_hits: dict[str, list[bool]] = {
        "uniform_attachment": [],
        "hub_preferential": [],
        "optional_relation": [],
        "one_to_one": [],
        "capacity_limited": [],
        "temporal_validity": [],
        "multi_parent_coupling": [],
        "bridge_pairing": [],
    }

    for database in databases:
        archetype = database.metadata.get("schema_archetype")
        archetype_counts[str(archetype) if isinstance(archetype, str) else "unlabeled"] += 1
        for spec in database.table_specs.values():
            role_counts[spec.role] += 1
        for fk in database.foreign_keys:
            attachment_counts[fk.mechanism.attachment] += 1
            existence_counts[fk.existence] += 1
            intent_counts[fk.semantic] += 1
        if len(database.foreign_keys) > 0:
            null_rates = [
                float(np.mean(database.foreign_key_null_masks[fk.key]))
                for fk in database.foreign_keys
            ]
            structural_null_rates.append(float(np.mean(null_rates)))
        feature_rates = [
            float(np.mean(mask))
            for mask in database.feature_missing_masks.values()
            if mask.size > 0
        ]
        if feature_rates:
            feature_missing_rates.append(float(np.mean(feature_rates)))

        regime_hits["uniform_attachment"].append(_has_uniform_attachment(database))
        regime_hits["hub_preferential"].append(_has_hub_attachment(database))
        regime_hits["optional_relation"].append(_has_optional_relation(database))
        regime_hits["one_to_one"].append(_has_one_to_one(database))
        regime_hits["capacity_limited"].append(_has_capacity_limited(database))
        regime_hits["temporal_validity"].append(_temporal_validity_holds(database))
        regime_hits["multi_parent_coupling"].append(_has_multi_parent_coupling(database))
        regime_hits["bridge_pairing"].append(_has_bridge_pairing(database))

    hit_rates = {name: float(np.mean(hits)) if hits else 0.0 for name, hits in regime_hits.items()}
    warnings: list[str] = []
    for name, rate in hit_rates.items():
        if rate < 0.5:
            warnings.append(f"low coverage for regime {name}: {rate:.2f}")

    return RDBStatisticsReport(
        sample_count=len(databases),
        archetype_counts=dict(archetype_counts),
        role_counts=dict(role_counts),
        attachment_counts=dict(attachment_counts),
        existence_counts=dict(existence_counts),
        intent_counts=dict(intent_counts),
        mean_tables=float(np.mean([len(db.table_specs) for db in databases])),
        mean_foreign_keys=float(np.mean([len(db.foreign_keys) for db in databases])),
        mean_structural_null_rate=float(np.mean(structural_null_rates))
        if structural_null_rates
        else 0.0,
        mean_feature_missing_rate=float(np.mean(feature_missing_rates))
        if feature_missing_rates
        else 0.0,
        regime_hits=hit_rates,
        warnings=tuple(warnings),
    )


def evaluate_regime_gate(
    databases: Sequence[RelationalDataset],
    thresholds: RDBTopologyRegimeThresholds | None = None,
) -> RDBTopologyRegimeGate:
    thresholds = thresholds or RDBTopologyRegimeThresholds()
    report = summarize_databases(databases)
    failures: list[str] = []
    for regime, rate in report.regime_hits.items():
        hit_count = int(round(rate * report.sample_count))
        if hit_count < thresholds.min_regime_hit_count:
            failures.append(f"{regime} hit count {hit_count} < {thresholds.min_regime_hit_count}")
        if rate < thresholds.min_regime_hit_rate:
            failures.append(f"{regime} hit rate {rate:.2f} < {thresholds.min_regime_hit_rate}")
    return RDBTopologyRegimeGate(
        passed=len(failures) == 0,
        failures=tuple(failures),
        regime_hits=report.regime_hits,
    )


def evaluate_expressivity_gate(
    databases: Sequence[RelationalDataset],
    thresholds: RDBTopologyRegimeThresholds | None = None,
) -> RDBTopologyRegimeGate:
    return evaluate_regime_gate(databases, thresholds=thresholds)


def compute_database_topology_stats(database: RelationalDataset) -> dict[str, object]:
    attachment_counts: Counter[str] = Counter()
    existence_counts: Counter[str] = Counter()
    edge_gini: list[float] = []
    edge_null_rates: list[float] = []
    mandatory_stats = database.metadata.get("mandatory_fk_stats", {})

    for fk in database.foreign_keys:
        attachment_counts[fk.mechanism.attachment] += 1
        existence_counts[fk.existence] += 1
        edge_gini.append(_fanout_gini(database, fk))
        edge_null_rates.append(float(np.mean(database.foreign_key_null_masks[fk.key])))

    return {
        "schema_archetype": database.metadata.get("schema_archetype"),
        "num_tables": len(database.table_specs),
        "num_foreign_keys": len(database.foreign_keys),
        "attachment_counts": dict(attachment_counts),
        "existence_counts": dict(existence_counts),
        "mean_edge_gini": float(np.mean(edge_gini)) if edge_gini else 0.0,
        "mean_structural_null_rate": float(np.mean(edge_null_rates)) if edge_null_rates else 0.0,
        "mandatory_fk_stats": mandatory_stats,
        "regime_hits": {
            "uniform_attachment": _has_uniform_attachment(database),
            "hub_preferential": _has_hub_attachment(database),
            "optional_relation": _has_optional_relation(database),
            "one_to_one": _has_one_to_one(database),
            "capacity_limited": _has_capacity_limited(database),
            "temporal_validity": _temporal_validity_holds(database),
            "multi_parent_coupling": _has_multi_parent_coupling(database),
            "bridge_pairing": _has_bridge_pairing(database),
        },
    }


def evaluate_calibration_gate_for_databases(
    databases: Sequence[RelationalDataset],
    bounds: MechanismInvariantBounds,
    *,
    allow_warnings: bool = True,
) -> CalibrationGateReport:
    return evaluate_calibration_gate(databases, bounds, allow_warnings=allow_warnings)


def generate_forced_regime_database(regime: str, seed: int = 0) -> RelationalDataset:
    hyper = MechanismHyperpriorConfig()
    role_grammar = RoleGrammarConfig()
    schema_archetype = SchemaArchetypeConfig()
    if regime == "uniform_attachment":
        hyper = MechanismHyperpriorConfig(
            forced_attachment="uniform",
            forced_existence="mandatory",
            hub_strength_range=(0.1, 0.3),
        )
    elif regime == "hub_preferential":
        hyper = MechanismHyperpriorConfig(
            forced_attachment="hub_preferential",
            forced_existence="mandatory",
            hub_strength_range=(2.5, 4.0),
        )
    elif regime == "optional_relation":
        hyper = MechanismHyperpriorConfig(forced_existence="optional", optional_existence_bias=-0.8)
    elif regime == "temporal_validity":
        role_grammar = replace(
            role_grammar,
            timestamp_probability_by_role={
                "entity": 1.0,
                "activity/event": 1.0,
                "dimension/lookup": 1.0,
                "bridge": 1.0,
                "snapshot/state": 1.0,
            },
        )
    elif regime == "bridge_pairing":
        schema_archetype = SchemaArchetypeConfig(forced_archetype="many-to-many")
    config = RDBPriorConfig(
        min_tables=5,
        max_tables=5,
        min_rows_per_table=64,
        max_rows_per_table=64,
        mechanism_hyperprior=hyper,
        role_grammar=role_grammar,
        schema_archetype=schema_archetype,
        optional_foreign_key_probability=1.0 if regime == "optional_relation" else 0.0,
        one_to_one_probability=1.0 if regime == "one_to_one" else 0.0,
        capacity_limited_probability=1.0
        if regime in ("capacity_limited",)
        else (0.0 if regime == "optional_relation" else 0.2),
        temporal_foreign_key_probability=1.0 if regime == "temporal_validity" else 0.0,
        multi_parent_probability=1.0
        if regime in ("multi_parent_coupling", "bridge_pairing")
        else 0.2,
        enable_many_to_many_motif=regime == "bridge_pairing",
        seed=seed,
    )
    for attempt in range(8):
        generator = RelationalPriorGenerator(replace(config, seed=seed + attempt))
        database = generator.sample_database()
        checks = {
            "uniform_attachment": _has_uniform_attachment,
            "hub_preferential": _has_hub_attachment,
            "optional_relation": _has_optional_relation,
            "one_to_one": _has_one_to_one,
            "capacity_limited": _has_capacity_limited,
            "temporal_validity": _temporal_validity_holds,
            "multi_parent_coupling": _has_multi_parent_coupling,
            "bridge_pairing": _has_bridge_pairing,
        }
        if regime in checks and checks[regime](database):
            return database
    raise RuntimeError(f"could not generate forced regime {regime!r} within 8 attempts")


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


def _has_uniform_attachment(database: RelationalDataset) -> bool:
    for fk in database.foreign_keys:
        if fk.mechanism.attachment == "uniform" and _fanout_gini(database, fk) < 0.35:
            return True
    return False


def _has_hub_attachment(database: RelationalDataset) -> bool:
    for fk in database.foreign_keys:
        if fk.mechanism.attachment == "hub_preferential" and _fanout_gini(database, fk) > 0.55:
            return True
    return False


def _has_optional_relation(database: RelationalDataset) -> bool:
    for fk in database.foreign_keys:
        if fk.existence in ("optional", "sparse"):
            null_rate = float(np.mean(database.foreign_key_null_masks[fk.key]))
            if 0.05 < null_rate < 0.5 and _latent_null_mi(database, fk) > 0.01:
                return True
    return False


def _has_one_to_one(database: RelationalDataset) -> bool:
    for fk in database.foreign_keys:
        if fk.cardinality == "one_to_one" or fk.mechanism.capacity_mode == "one_to_one":
            values = database.column_values(fk.child_table, fk.child_column)
            parent_rows = values[np.isfinite(values)].astype(np.int64)
            counts = np.bincount(
                parent_rows, minlength=database.table_specs[fk.parent_table].row_count
            )
            if int(counts.max(initial=0)) <= 1:
                return True
    return False


def _has_capacity_limited(database: RelationalDataset) -> bool:
    for fk in database.foreign_keys:
        if fk.capacity is not None:
            values = database.column_values(fk.child_table, fk.child_column)
            parent_rows = values[np.isfinite(values)].astype(np.int64)
            counts = np.bincount(
                parent_rows, minlength=database.table_specs[fk.parent_table].row_count
            )
            if int(counts.max(initial=0)) <= fk.capacity:
                return True
    return False


def _temporal_validity_holds(database: RelationalDataset) -> bool:
    for fk in database.foreign_keys:
        if not fk.temporal:
            continue
        child_times = database.column_values(fk.child_table, "timestamp")
        parent_times = database.column_values(fk.parent_table, "timestamp")
        values = database.column_values(fk.child_table, fk.child_column)
        for child_row, parent_value in enumerate(values):
            if np.isfinite(parent_value):
                if float(parent_times[int(parent_value)]) > float(child_times[child_row]) + 1e-6:
                    return False
    return any(fk.temporal for fk in database.foreign_keys)


def _has_multi_parent_coupling(database: RelationalDataset) -> bool:
    groups: dict[str, list[ForeignKeySpec]] = {}
    for fk in database.foreign_keys:
        if fk.multi_parent_group is not None:
            groups.setdefault(fk.multi_parent_group, []).append(fk)
    for group in groups.values():
        if len(group) < 2:
            continue
        first, second = group[0], group[1]
        a = database.column_values(first.child_table, first.child_column)
        b = database.column_values(second.child_table, second.child_column)
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 8:
            continue
        observed = _mutual_information(a[mask].astype(np.int64), b[mask].astype(np.int64))
        perm_seed = zlib.crc32(first.key.encode("utf-8"))
        perm = np.random.default_rng(perm_seed).permutation(int(mask.sum()))
        shuffled = _mutual_information(a[mask].astype(np.int64), b[mask].astype(np.int64)[perm])
        if shuffled <= 1e-12:
            return observed > 0.02
        if observed > shuffled + 0.01:
            return True
    return False


def _has_bridge_pairing(database: RelationalDataset) -> bool:
    for fk in database.foreign_keys:
        if fk.semantic != "bridge_pairs_entities":
            continue
        values = database.column_values(fk.child_table, fk.child_column)
        if np.isfinite(values).sum() > 0:
            return True
    bridge_tables = [spec for spec in database.table_specs.values() if spec.role == "bridge"]
    if not bridge_tables:
        return False
    for table in bridge_tables:
        fks = [fk for fk in database.foreign_keys if fk.child_table == table.name]
        if len(fks) >= 2:
            parents = set()
            for fk in fks:
                values = database.column_values(fk.child_table, fk.child_column)
                parents.update(values[np.isfinite(values)].astype(np.int64).tolist())
            return len(parents) > 1
    return False


def _latent_null_mi(database: RelationalDataset, fk: ForeignKeySpec) -> float:
    null_mask = database.foreign_key_null_masks[fk.key]
    latent = database.row_embeddings[fk.child_table][:, 0]
    discretized = np.digitize(latent, bins=np.quantile(latent, [0.25, 0.5, 0.75]))
    return _mutual_information(discretized.astype(np.int64), null_mask.astype(np.int64))


def _mutual_information(x: NDArray[np.int64], y: NDArray[np.int64]) -> float:
    if len(x) == 0:
        return 0.0
    x_values, x_inverse = np.unique(x, return_inverse=True)
    y_values, y_inverse = np.unique(y, return_inverse=True)
    joint = np.zeros((len(x_values), len(y_values)), dtype=np.float64)
    np.add.at(joint, (x_inverse, y_inverse), 1.0)
    joint = joint / float(len(x))
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = joint > 0.0
    return float(np.sum(joint[mask] * np.log(joint[mask] / expected[mask])))
