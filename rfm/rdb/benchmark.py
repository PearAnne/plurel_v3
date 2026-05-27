from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from rfm.benchmark import (
    _prepare_probe_features,
    _ridge_classify,
    _ridge_regress,
    make_dataset_from_arrays,
)
from rfm.rdb.config import MechanismHyperpriorConfig, RDBPriorConfig, SchemaArchetypeConfig
from rfm.rdb.generator import RelationalPriorGenerator
from rfm.rdb.types import ForeignKeySpec, RelationalDataset, RelationalTargetFamily, RelationalTask
from rfm.types import SyntheticDataset, TaskType

ProbeModel = Literal[
    "focal_only",
    "joined_flat",
    "joined_plus_interactions",
    "topology_only",
    "joined_plus_topology",
    "relational_mp",
    "independent_single_table",
    "joined_fk_shuffled",
    "joined_parent_permuted",
    "joined_same_schema_no_fk",
    "relational_fk_shuffled",
    "relational_parent_permuted",
    "relational_same_schema_no_fk",
]
RelationalControl = Literal["fk_shuffled", "parent_permuted", "same_schema_no_fk"]
RelationalView = Literal[
    "joined_plus_interactions", "topology_only", "joined_plus_topology", "relational_mp"
]
TaskFilterMode = Literal["validity_only", "signal_conditioned"]
BenchmarkSuite = Literal["base_balanced", "many_to_many", "temporal_history"]
TARGET_FAMILIES: tuple[RelationalTargetFamily, ...] = (
    "local_only",
    "parent_feature",
    "parent_child_interaction",
    "multi_parent",
    "topology_driven",
)
TASK_TYPES: tuple[TaskType, ...] = ("classification", "regression")


@dataclass(frozen=True)
class RelationalProbeResult:
    model: ProbeModel
    task_count: int
    score_mean: float | None
    classification_accuracy_mean: float | None
    regression_r2_mean: float | None
    gain_vs_focal: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RelationalBenchmarkReport:
    results: dict[str, RelationalProbeResult]
    gain_joined: float | None
    gain_relational: float | None
    gain_vs_independent: float | None
    control_results: dict[str, RelationalProbeResult] | None = None
    family_results: dict[str, dict[str, object]] | None = None
    sampling_summary: dict[str, object] | None = None
    stability_results: dict[str, object] | None = None
    generation_health: dict[str, object] | None = None
    schema_calibration: dict[str, object] | None = None
    benchmark_gate: dict[str, object] | None = None
    task_records: list[dict[str, object]] | None = None
    task_level_statistics: dict[str, object] | None = None
    evaluation_metadata: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "results": {name: result.to_dict() for name, result in self.results.items()},
            "gain_joined": self.gain_joined,
            "gain_relational": self.gain_relational,
            "gain_vs_independent": self.gain_vs_independent,
            "control_results": None
            if self.control_results is None
            else {name: result.to_dict() for name, result in self.control_results.items()},
            "family_results": self.family_results,
            "sampling_summary": self.sampling_summary,
            "stability_results": self.stability_results,
            "generation_health": self.generation_health,
            "schema_calibration": self.schema_calibration,
            "benchmark_gate": self.benchmark_gate,
            "task_records": self.task_records,
            "task_level_statistics": self.task_level_statistics,
            "evaluation_metadata": self.evaluation_metadata,
        }
        return payload


def _task_row_order(task: RelationalTask) -> NDArray[np.int64]:
    return np.concatenate([task.train_indices, task.test_indices]).astype(np.int64)


def _task_target_values(database: RelationalDataset, task: RelationalTask) -> NDArray[np.float32]:
    if task.target_values is not None:
        return task.target_values
    return database.column_values(task.target_table, task.target_column)


def _focal_feature_indices(database: RelationalDataset, task: RelationalTask) -> list[int]:
    spec = database.table_specs[task.target_table]
    indices: list[int] = []
    for column in spec.feature_columns:
        if column.name == task.target_column:
            continue
        indices.append(spec.column_index(column.name))
    if not indices:
        if spec.timestamp_column is None:
            raise ValueError(f"table {task.target_table!r} has no focal features")
        indices.append(spec.column_index(spec.timestamp_column))
    return indices


def build_focal_dataset(database: RelationalDataset, task: RelationalTask) -> SyntheticDataset:
    table = database.tables[task.target_table]
    order = _task_row_order(task)
    feature_indices = _focal_feature_indices(database, task)
    x = table[order][:, feature_indices]
    y = _task_target_values(database, task)[order]
    return make_dataset_from_arrays(
        x=x,
        y=y,
        source_name=f"{task.target_table}.{task.target_column}",
        prior_type="rdb",
        train_size=len(task.train_indices),
        task_type=task.task_type,
        shuffle=False,
    )


def build_joined_dataset(database: RelationalDataset, task: RelationalTask) -> SyntheticDataset:
    return _build_joined_dataset(database, task, control=None, seed=0)


def build_fk_shuffled_dataset(
    database: RelationalDataset, task: RelationalTask, seed: int
) -> SyntheticDataset:
    return _build_joined_dataset(database, task, control="fk_shuffled", seed=seed)


def build_parent_permuted_dataset(
    database: RelationalDataset, task: RelationalTask, seed: int
) -> SyntheticDataset:
    return _build_joined_dataset(database, task, control="parent_permuted", seed=seed)


def build_same_schema_no_fk_dataset(
    database: RelationalDataset, task: RelationalTask, seed: int
) -> SyntheticDataset:
    return _build_joined_dataset(database, task, control="same_schema_no_fk", seed=seed)


def _build_joined_dataset(
    database: RelationalDataset,
    task: RelationalTask,
    control: RelationalControl | None,
    seed: int,
) -> SyntheticDataset:
    table = database.tables[task.target_table]
    order = _task_row_order(task)
    feature_indices = _focal_feature_indices(database, task)
    joined_blocks = [table[order][:, feature_indices]]
    rng = np.random.default_rng(seed)
    for fk in database.foreign_keys:
        if fk.child_table != task.target_table:
            continue
        parent_spec = database.table_specs[fk.parent_table]
        parent_feature_indices = [
            parent_spec.column_index(column.name) for column in parent_spec.feature_columns
        ]
        if len(parent_feature_indices) == 0:
            continue
        parent_table = database.tables[fk.parent_table]
        fk_values = database.column_values(task.target_table, fk.child_column)[order]
        if control == "fk_shuffled":
            fk_values = rng.permutation(fk_values)
        joined = np.full((len(order), len(parent_feature_indices)), np.nan, dtype=np.float32)
        for row_idx, parent_value in enumerate(fk_values):
            if control == "same_schema_no_fk":
                parent_row = int(rng.integers(0, parent_table.shape[0]))
                joined[row_idx] = parent_table[parent_row, parent_feature_indices]
            elif np.isfinite(parent_value):
                joined[row_idx] = parent_table[int(parent_value), parent_feature_indices]
        if control == "parent_permuted":
            joined = joined[rng.permutation(joined.shape[0])]
        joined_blocks.append(joined.astype(np.float32))
    x = np.concatenate(joined_blocks, axis=1)
    y = _task_target_values(database, task)[order]
    return make_dataset_from_arrays(
        x=x,
        y=y,
        source_name=f"{task.target_table}.{task.target_column}.{control or 'joined'}",
        prior_type="rdb",
        train_size=len(task.train_indices),
        task_type=task.task_type,
        shuffle=False,
    )


def build_relational_dataset(database: RelationalDataset, task: RelationalTask) -> SyntheticDataset:
    return _build_relational_view_dataset(
        database, task, view="relational_mp", control=None, seed=0
    )


def build_joined_plus_interactions_dataset(
    database: RelationalDataset, task: RelationalTask
) -> SyntheticDataset:
    return _build_relational_view_dataset(
        database, task, view="joined_plus_interactions", control=None, seed=0
    )


def build_topology_dataset(database: RelationalDataset, task: RelationalTask) -> SyntheticDataset:
    return _build_relational_view_dataset(
        database, task, view="topology_only", control=None, seed=0
    )


def build_joined_plus_topology_dataset(
    database: RelationalDataset, task: RelationalTask
) -> SyntheticDataset:
    return _build_relational_view_dataset(
        database, task, view="joined_plus_topology", control=None, seed=0
    )


def build_relational_control_dataset(
    database: RelationalDataset,
    task: RelationalTask,
    control: RelationalControl,
    seed: int,
) -> SyntheticDataset:
    return _build_relational_view_dataset(
        database, task, view="relational_mp", control=control, seed=seed
    )


def _build_relational_view_dataset(
    database: RelationalDataset,
    task: RelationalTask,
    view: RelationalView,
    control: RelationalControl | None,
    seed: int,
) -> SyntheticDataset:
    table = database.tables[task.target_table]
    order = _task_row_order(task)
    feature_indices = _focal_feature_indices(database, task)
    x_focal = table[order][:, feature_indices]
    joined_blocks = [x_focal]
    message_dim = 8
    messages = np.zeros((len(order), message_dim), dtype=np.float32)
    parent_feature_blocks: list[NDArray[np.float32]] = []
    effective_values = _controlled_fk_values(database, control=control, seed=seed)
    for fk in database.foreign_keys:
        if fk.child_table != task.target_table:
            continue
        fk_values = effective_values[fk.key][order]
        parent_spec = database.table_specs[fk.parent_table]
        parent_feature_indices = [
            parent_spec.column_index(column.name) for column in parent_spec.feature_columns
        ]
        if parent_feature_indices:
            parent_table = database.tables[fk.parent_table]
            joined = np.full((len(order), len(parent_feature_indices)), np.nan, dtype=np.float32)
            for row_idx, parent_value in enumerate(fk_values):
                if np.isfinite(parent_value):
                    joined[row_idx] = parent_table[int(parent_value), parent_feature_indices]
            joined_blocks.append(joined.astype(np.float32))
            parent_feature_blocks.append(joined.astype(np.float32))
        parent_latents = database.row_embeddings[fk.parent_table]
        for row_idx, parent_value in enumerate(fk_values):
            if np.isfinite(parent_value):
                latent = parent_latents[int(parent_value)]
                messages[row_idx, : min(message_dim, latent.shape[0])] = latent[:message_dim]
    interaction_blocks = _relational_interaction_blocks(x_focal, parent_feature_blocks)
    topology = _relational_topology_features(database, task, order, effective_values)
    if view == "joined_plus_interactions":
        x = np.concatenate([*joined_blocks, *interaction_blocks], axis=1)
    elif view == "topology_only":
        x = topology
    elif view == "joined_plus_topology":
        x = np.concatenate([*joined_blocks, topology], axis=1)
    else:
        x = np.concatenate([*joined_blocks, messages, topology, *interaction_blocks], axis=1)
    y = _task_target_values(database, task)[order]
    return make_dataset_from_arrays(
        x=x,
        y=y,
        source_name=f"{task.target_table}.{task.target_column}.{view}.{control or 'original'}",
        prior_type="rdb",
        train_size=len(task.train_indices),
        task_type=task.task_type,
        shuffle=False,
    )


def _relational_interaction_blocks(
    x_focal: NDArray[np.float32],
    parent_feature_blocks: Sequence[NDArray[np.float32]],
) -> list[NDArray[np.float32]]:
    blocks: list[NDArray[np.float32]] = []
    if x_focal.shape[1] > 0:
        focal = np.nan_to_num(x_focal, nan=0.0).astype(np.float32)
        for parent_features in parent_feature_blocks:
            parent = np.nan_to_num(parent_features, nan=0.0).astype(np.float32)
            interactions = focal[:, :, None] * parent[:, None, :]
            blocks.append(interactions.reshape(focal.shape[0], -1).astype(np.float32))
    for left_idx in range(len(parent_feature_blocks)):
        for right_idx in range(left_idx + 1, len(parent_feature_blocks)):
            left = np.nan_to_num(parent_feature_blocks[left_idx], nan=0.0).astype(np.float32)
            right = np.nan_to_num(parent_feature_blocks[right_idx], nan=0.0).astype(np.float32)
            interactions = left[:, :, None] * right[:, None, :]
            blocks.append(interactions.reshape(left.shape[0], -1).astype(np.float32))
    return blocks


def _relational_topology_features(
    database: RelationalDataset,
    task: RelationalTask,
    order: NDArray[np.int64],
    effective_values: Mapping[str, NDArray[np.float32]] | None = None,
) -> NDArray[np.float32]:
    features: list[NDArray[np.float32]] = []
    row_count = len(order)
    for fk in database.foreign_keys:
        if fk.child_table == task.target_table:
            all_values = (
                database.column_values(fk.child_table, fk.child_column)
                if effective_values is None
                else effective_values[fk.key]
            )
            values = all_values[order]
            exists = np.isfinite(values).astype(np.float32)[:, None]
            parent_counts = _edge_parent_counts(database, fk, all_values)
            joined_fanout = np.zeros(row_count, dtype=np.float32)
            mask = np.isfinite(values)
            if mask.any():
                joined_fanout[mask] = np.log1p(parent_counts[values[mask].astype(np.int64)]).astype(
                    np.float32
                )
            features.append(exists)
            features.append(_standardize_feature(joined_fanout)[:, None])
        if fk.parent_table == task.target_table:
            values = None if effective_values is None else effective_values[fk.key]
            parent_counts = _edge_parent_counts(database, fk, values)
            features.append(
                _standardize_feature(np.log1p(parent_counts[order]).astype(np.float32))[:, None]
            )
    if not features:
        return np.zeros((row_count, 1), dtype=np.float32)
    return np.concatenate(features, axis=1).astype(np.float32)


def _edge_parent_counts(
    database: RelationalDataset,
    fk: ForeignKeySpec,
    values: NDArray[np.float32] | None = None,
) -> NDArray[np.float64]:
    if values is None:
        values = database.column_values(fk.child_table, fk.child_column)
    parent_count = database.table_specs[fk.parent_table].row_count
    parent_rows = values[np.isfinite(values)].astype(np.int64)
    return np.bincount(parent_rows, minlength=parent_count).astype(np.float64)


def _controlled_fk_values(
    database: RelationalDataset,
    control: RelationalControl | None,
    seed: int,
) -> dict[str, NDArray[np.float32]]:
    rng = np.random.default_rng(seed)
    effective: dict[str, NDArray[np.float32]] = {}
    for fk in database.foreign_keys:
        values = database.column_values(fk.child_table, fk.child_column).astype(
            np.float32, copy=True
        )
        if control in ("fk_shuffled", "parent_permuted"):
            values = rng.permutation(values).astype(np.float32, copy=False)
        elif control == "same_schema_no_fk":
            parent_rows = database.table_specs[fk.parent_table].row_count
            values = rng.integers(0, parent_rows, size=len(values)).astype(np.float32)
        effective[fk.key] = values
    return effective


def _standardize_feature(values: NDArray[np.float32]) -> NDArray[np.float32]:
    cleaned = np.nan_to_num(values.astype(np.float64), nan=0.0)
    std = float(np.std(cleaned))
    if std <= 1e-12 or not np.isfinite(std):
        return (cleaned - float(np.mean(cleaned))).astype(np.float32)
    return ((cleaned - float(np.mean(cleaned))) / std).astype(np.float32)


def build_independent_single_table_dataset(
    database: RelationalDataset, task: RelationalTask, seed: int
) -> SyntheticDataset:
    focal = build_focal_dataset(database, task)
    num_tables = max(len(database.table_specs), 1)
    total_features = focal.x.shape[1] * num_tables
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(focal.meta.num_rows, total_features)).astype(np.float32)
    y = focal.y[rng.permutation(focal.meta.num_rows)]
    return make_dataset_from_arrays(
        x=x,
        y=y,
        source_name="independent_single_table",
        prior_type="scm",
        train_size=focal.meta.train_size,
        task_type=task.task_type,
        shuffle=False,
    )


def evaluate_relational_probe(dataset: SyntheticDataset, ridge_alpha: float = 1.0) -> float:
    features = _prepare_probe_features(dataset)
    train_size = dataset.meta.train_size
    x_train = features[:train_size]
    x_test = features[train_size:]
    if dataset.meta.task_type == "classification":
        y = dataset.y.astype(np.int64)
        classes = np.unique(y[:train_size])
        pred = _ridge_classify(x_train, y[:train_size], x_test, classes, ridge_alpha)
        return float(np.mean(y[train_size:] == pred))
    y_float = dataset.y.astype(np.float64)
    pred = _ridge_regress(x_train, y_float[:train_size], x_test, ridge_alpha)
    variance = float(np.sum((y_float[train_size:] - float(np.mean(y_float[train_size:]))) ** 2))
    residual = float(np.sum((pred - y_float[train_size:]) ** 2))
    return 1.0 - residual / variance if variance > 1e-12 else 0.0


def _empty_score_map() -> dict[ProbeModel, list[float]]:
    return {
        "focal_only": [],
        "joined_flat": [],
        "joined_plus_interactions": [],
        "topology_only": [],
        "joined_plus_topology": [],
        "relational_mp": [],
        "independent_single_table": [],
        "joined_fk_shuffled": [],
        "joined_parent_permuted": [],
        "joined_same_schema_no_fk": [],
        "relational_fk_shuffled": [],
        "relational_parent_permuted": [],
        "relational_same_schema_no_fk": [],
    }


def _task_family(task: RelationalTask) -> str:
    if task.target_spec is not None:
        return task.target_spec.target_family
    return f"legacy:{task.target_dependency_kind}"


def _evaluate_task_scores(
    database: RelationalDataset,
    task: RelationalTask,
    control_seed: int,
    ridge_alpha: float,
) -> dict[ProbeModel, float]:
    datasets = {
        "focal_only": build_focal_dataset(database, task),
        "joined_flat": build_joined_dataset(database, task),
        "joined_plus_interactions": build_joined_plus_interactions_dataset(database, task),
        "topology_only": build_topology_dataset(database, task),
        "joined_plus_topology": build_joined_plus_topology_dataset(database, task),
        "relational_mp": build_relational_dataset(database, task),
        "independent_single_table": build_independent_single_table_dataset(
            database, task, seed=control_seed
        ),
        "joined_fk_shuffled": build_fk_shuffled_dataset(database, task, seed=control_seed + 1000),
        "joined_parent_permuted": build_parent_permuted_dataset(
            database, task, seed=control_seed + 2000
        ),
        "joined_same_schema_no_fk": build_same_schema_no_fk_dataset(
            database, task, seed=control_seed + 3000
        ),
        "relational_fk_shuffled": build_relational_control_dataset(
            database, task, control="fk_shuffled", seed=control_seed + 4000
        ),
        "relational_parent_permuted": build_relational_control_dataset(
            database, task, control="parent_permuted", seed=control_seed + 5000
        ),
        "relational_same_schema_no_fk": build_relational_control_dataset(
            database, task, control="same_schema_no_fk", seed=control_seed + 6000
        ),
    }
    return {
        model_name: evaluate_relational_probe(dataset, ridge_alpha=ridge_alpha)
        for model_name, dataset in datasets.items()
    }


def _append_task_scores(
    scores: dict[ProbeModel, list[float]],
    score_task_types: dict[ProbeModel, list[TaskType]],
    family_scores: dict[str, dict[ProbeModel, list[float]]],
    task: RelationalTask,
    task_scores: dict[ProbeModel, float],
) -> None:
    family = _task_family(task)
    family_scores.setdefault(family, {name: [] for name in scores})
    for model_name, score in task_scores.items():
        scores[model_name].append(score)
        score_task_types[model_name].append(task.task_type)
        family_scores[family][model_name].append(score)


def _benchmark_sample_record(
    database: RelationalDataset,
    task: RelationalTask,
    sample_index: int,
    seed: int,
    attempts: int,
    task_rejection_histogram: Mapping[str, int] | None = None,
    requested_family: RelationalTargetFamily | None = None,
    requested_task_type: TaskType | None = None,
) -> dict[str, object]:
    database_stats = database.metadata.get("database_sampling_stats", {})
    mandatory_stats = database.metadata.get("mandatory_fk_stats", {})
    preflight_stats = database.metadata.get("feasibility_preflight", {})
    if not isinstance(database_stats, dict):
        database_stats = {}
    if not isinstance(mandatory_stats, dict):
        mandatory_stats = {}
    if not isinstance(preflight_stats, dict):
        preflight_stats = {}
    family = _task_family(task)
    difficulty_metrics = None
    if task.target_spec is not None and task.target_spec.difficulty_metrics is not None:
        difficulty_metrics = asdict(task.target_spec.difficulty_metrics)
    archetype = database.metadata.get("schema_archetype")
    bridge_present = any(spec.role == "bridge" for spec in database.table_specs.values())
    snapshot_present = any(spec.role == "snapshot/state" for spec in database.table_specs.values())
    return {
        "sample_id": f"{family}::{task.task_type}::{sample_index}::{seed}",
        "family": family,
        "requested_family": requested_family,
        "task_type": task.task_type,
        "requested_task_type": requested_task_type,
        "sample_index": sample_index,
        "seed": seed,
        "attempts": attempts,
        "metric": "accuracy" if task.task_type == "classification" else "r2",
        "schema_archetype": archetype if isinstance(archetype, str) else "unlabeled",
        "motifs": {
            "bridge": bridge_present,
            "snapshot": snapshot_present,
            "bridge_applicable": archetype == "many-to-many",
            "snapshot_applicable": archetype == "temporal-history",
        },
        "difficulty_metrics": difficulty_metrics,
        "task_rejection_histogram": dict(task_rejection_histogram or {}),
        "database_attempts": _int_value(database_stats.get("attempts"), default=1),
        "database_retries": _int_value(database_stats.get("retries"), default=0),
        "mandatory_fk_total": _int_value(mandatory_stats.get("mandatory_fk_total"), default=0),
        "mandatory_fk_unsatisfied": _int_value(
            mandatory_stats.get("mandatory_fk_unsatisfied"), default=0
        ),
        "mandatory_fk_forced_null": _int_value(
            mandatory_stats.get("mandatory_fk_forced_null"), default=0
        ),
        "mandatory_fk_backoff_count": _int_value(
            mandatory_stats.get("mandatory_fk_backoff_count"), default=0
        ),
        "mandatory_fk_timestamp_resample_count": _int_value(
            mandatory_stats.get("mandatory_fk_timestamp_resample_count"),
            default=0,
        ),
        "multi_parent_candidate_empty_count": _int_value(
            mandatory_stats.get("multi_parent_candidate_empty_count"),
            default=0,
        ),
        "joint_sampler_backoff_count": _int_value(
            mandatory_stats.get("joint_sampler_backoff_count"),
            default=0,
        ),
        "joint_sampler_independent_fallback_count": _int_value(
            mandatory_stats.get("joint_sampler_independent_fallback_count"),
            default=0,
        ),
        "checked_edges": _int_value(
            preflight_stats.get("checked_edges"), default=len(database.foreign_keys)
        ),
        "edge_adjustment_count": _int_value(
            preflight_stats.get("edge_adjustment_count"), default=0
        ),
        "edge_downgrade_count": _int_value(preflight_stats.get("edge_downgrade_count"), default=0),
        "preflight_reason_histogram": _string_int_map(preflight_stats.get("reason_histogram")),
        "preflight_action_histogram": _string_int_map(preflight_stats.get("action_histogram")),
        "split_kind": task.split_kind,
        "target_table": task.target_table,
        "target_column": task.target_column,
    }


def _schema_record(database: RelationalDataset) -> dict[str, object]:
    role_counts = Counter(spec.role for spec in database.table_specs.values())
    timestamp_counts = Counter(
        spec.role for spec in database.table_specs.values() if spec.has_timestamp
    )
    root_tables = _root_table_count(database)
    archetype = database.metadata.get("schema_archetype")
    bridge_present = any(spec.role == "bridge" for spec in database.table_specs.values())
    snapshot_present = any(spec.role == "snapshot/state" for spec in database.table_specs.values())
    structural_null_rates = [
        float(np.mean(database.foreign_key_null_masks[fk.key])) for fk in database.foreign_keys
    ]
    return {
        "archetype": archetype if isinstance(archetype, str) else "unlabeled",
        "table_count": len(database.table_specs),
        "edge_count": len(database.foreign_keys),
        "role_counts": dict(role_counts),
        "timestamp_counts_by_role": dict(timestamp_counts),
        "temporal_fk_count": sum(1 for fk in database.foreign_keys if fk.temporal),
        "bridge_present": bridge_present,
        "snapshot_present": snapshot_present,
        "bridge_motif_violation": bridge_present and archetype != "many-to-many",
        "snapshot_motif_violation": snapshot_present and archetype != "temporal-history",
        "structural_null_rate": (
            float(np.mean(structural_null_rates)) if structural_null_rates else None
        ),
        **_foreign_key_constraint_record(database),
        "edge_density": len(database.foreign_keys) / max(float(len(database.table_specs)), 1.0),
        "dag_depth": _dag_depth(database),
        "root_table_ratio": root_tables / max(float(len(database.table_specs)), 1.0),
    }


def _foreign_key_constraint_record(database: RelationalDataset) -> dict[str, int]:
    counters = {
        "mandatory_fk_count": 0,
        "mandatory_null_violation_count": 0,
        "temporal_fk_count_for_validation": 0,
        "temporal_violation_count": 0,
        "one_to_one_fk_count": 0,
        "one_to_one_violation_count": 0,
        "capacity_fk_count": 0,
        "capacity_violation_count": 0,
    }
    for fk in database.foreign_keys:
        values = database.column_values(fk.child_table, fk.child_column)
        finite_mask = np.isfinite(values)
        parent_rows = values[finite_mask].astype(np.int64)
        if fk.existence == "mandatory":
            counters["mandatory_fk_count"] += 1
            if not bool(np.all(finite_mask)):
                counters["mandatory_null_violation_count"] += 1
        if fk.temporal:
            counters["temporal_fk_count_for_validation"] += 1
            child_timestamp = database.table_specs[fk.child_table].timestamp_column
            parent_timestamp = database.table_specs[fk.parent_table].timestamp_column
            if child_timestamp is None or parent_timestamp is None:
                counters["temporal_violation_count"] += 1
            elif parent_rows.size > 0:
                child_times = database.column_values(fk.child_table, child_timestamp)[finite_mask]
                parent_times = database.column_values(fk.parent_table, parent_timestamp)[
                    parent_rows
                ]
                if bool(np.any(parent_times > child_times + 1e-6)):
                    counters["temporal_violation_count"] += 1
        if fk.cardinality == "one_to_one" or fk.mechanism.capacity_mode == "one_to_one":
            counters["one_to_one_fk_count"] += 1
            fanout = np.bincount(
                parent_rows, minlength=database.table_specs[fk.parent_table].row_count
            )
            if int(fanout.max(initial=0)) > 1:
                counters["one_to_one_violation_count"] += 1
        if fk.capacity is not None:
            counters["capacity_fk_count"] += 1
            fanout = np.bincount(
                parent_rows, minlength=database.table_specs[fk.parent_table].row_count
            )
            if int(fanout.max(initial=0)) > fk.capacity:
                counters["capacity_violation_count"] += 1
    return counters


def _summarize_relational_scores(
    scores: dict[ProbeModel, list[float]],
    score_task_types: dict[ProbeModel, list[TaskType]],
    family_scores: dict[str, dict[ProbeModel, list[float]]],
    sampling_summary: dict[str, object] | None = None,
    stability_results: dict[str, object] | None = None,
    sample_records: Sequence[dict[str, object]] | None = None,
    schema_records: Sequence[dict[str, object]] | None = None,
    task_records: list[dict[str, object]] | None = None,
    evaluation_metadata: dict[str, object] | None = None,
    suite: BenchmarkSuite = "base_balanced",
) -> RelationalBenchmarkReport:
    def typed_mean(model_name: ProbeModel, task_type: TaskType) -> float | None:
        values = [
            score
            for score, score_task_type in zip(scores[model_name], score_task_types[model_name])
            if score_task_type == task_type
        ]
        return float(np.mean(values)) if values else None

    results: dict[str, RelationalProbeResult] = {}
    for model_name, model_scores in {
        key: scores[key]
        for key in (
            "focal_only",
            "joined_flat",
            "joined_plus_interactions",
            "topology_only",
            "joined_plus_topology",
            "relational_mp",
            "independent_single_table",
        )
    }.items():
        metric = float(np.mean(model_scores)) if model_scores else None
        focal_metric = float(np.mean(scores["focal_only"])) if scores["focal_only"] else None
        gain = None if metric is None or focal_metric is None else metric - focal_metric
        results[model_name] = RelationalProbeResult(
            model=model_name,
            task_count=len(model_scores),
            score_mean=metric,
            classification_accuracy_mean=typed_mean(model_name, "classification"),
            regression_r2_mean=typed_mean(model_name, "regression"),
            gain_vs_focal=gain,
        )
    control_results: dict[str, RelationalProbeResult] = {}
    for model_name in (
        "joined_fk_shuffled",
        "joined_parent_permuted",
        "joined_same_schema_no_fk",
        "relational_fk_shuffled",
        "relational_parent_permuted",
        "relational_same_schema_no_fk",
    ):
        model_scores = scores[model_name]
        metric = float(np.mean(model_scores)) if model_scores else None
        focal_metric = float(np.mean(scores["focal_only"])) if scores["focal_only"] else None
        gain = None if metric is None or focal_metric is None else metric - focal_metric
        control_results[model_name] = RelationalProbeResult(
            model=model_name,
            task_count=len(model_scores),
            score_mean=metric,
            classification_accuracy_mean=typed_mean(model_name, "classification"),
            regression_r2_mean=typed_mean(model_name, "regression"),
            gain_vs_focal=gain,
        )
    family_results: dict[str, dict[str, object]] = {}
    for family, by_model in family_scores.items():
        focal_family = float(np.mean(by_model["focal_only"])) if by_model["focal_only"] else 0.0
        family_results[family] = {
            model_name: float(np.mean(model_scores)) if model_scores else 0.0
            for model_name, model_scores in by_model.items()
        }
        family_results[family]["gain_joined"] = family_results[family]["joined_flat"] - focal_family
        family_results[family]["gain_relational"] = (
            family_results[family]["relational_mp"] - focal_family
        )
        family_results[family]["task_count"] = len(by_model["focal_only"])

    if sample_records is not None:
        _attach_family_generation_stats(family_results, sample_records)

    focal = results["focal_only"].score_mean
    joined = results["joined_flat"].score_mean
    relational = results["relational_mp"].score_mean
    independent = results["independent_single_table"].score_mean
    gain_joined = None if focal is None or joined is None else joined - focal
    gain_relational = None if focal is None or relational is None else relational - focal
    gain_vs_independent = (
        None if relational is None or independent is None else relational - independent
    )
    generation_health = (
        None if sample_records is None else _generation_health_summary(sample_records)
    )
    schema_calibration = (
        None if schema_records is None else _schema_calibration_summary(schema_records)
    )
    benchmark_gate = _benchmark_gate_summary(
        generation_health=generation_health,
        schema_calibration=schema_calibration,
        family_results=family_results,
        suite=suite,
    )
    task_level_statistics = None if task_records is None else _task_level_statistics(task_records)
    return RelationalBenchmarkReport(
        results=results,
        gain_joined=gain_joined,
        gain_relational=gain_relational,
        gain_vs_independent=gain_vs_independent,
        control_results=control_results,
        family_results=family_results,
        sampling_summary=sampling_summary,
        stability_results=stability_results,
        generation_health=generation_health,
        schema_calibration=schema_calibration,
        benchmark_gate=benchmark_gate,
        task_records=task_records,
        task_level_statistics=task_level_statistics,
        evaluation_metadata=evaluation_metadata,
    )


def _task_level_statistics(task_records: list[dict[str, object]]) -> dict[str, object]:
    by_family: dict[str, list[dict[str, object]]] = {}
    by_task_type: dict[str, list[dict[str, object]]] = {}
    by_cell: dict[str, list[dict[str, object]]] = {}
    by_attempt_bucket: dict[str, list[dict[str, object]]] = {"1-3": [], "4-10": [], ">10": []}
    for record in task_records:
        family = str(record["family"])
        task_type = str(record["task_type"])
        by_family.setdefault(family, []).append(record)
        by_task_type.setdefault(task_type, []).append(record)
        by_cell.setdefault(_cell_key(family, task_type), []).append(record)
        attempts = _int_value(record.get("attempts"), default=1)
        bucket = "1-3" if attempts <= 3 else "4-10" if attempts <= 10 else ">10"
        by_attempt_bucket[bucket].append(record)
    return {
        "overall": _task_group_statistics(task_records),
        "by_family": {
            key: _task_group_statistics(value) for key, value in sorted(by_family.items())
        },
        "by_task_type": {
            key: _task_group_statistics(value) for key, value in sorted(by_task_type.items())
        },
        "by_cell": {key: _task_group_statistics(value) for key, value in sorted(by_cell.items())},
        "by_attempt_bucket": {
            key: _task_group_statistics(value) for key, value in by_attempt_bucket.items() if value
        },
    }


def _task_group_statistics(task_records: Sequence[dict[str, object]]) -> dict[str, object]:
    model_scores: dict[str, list[float]] = {}
    for record in task_records:
        scores = record.get("scores")
        if not isinstance(scores, dict):
            continue
        for model_name, value in scores.items():
            if isinstance(value, float):
                model_scores.setdefault(str(model_name), []).append(value)
    return {
        "task_count": len(task_records),
        "score_means": {
            model_name: float(np.mean(values))
            for model_name, values in sorted(model_scores.items())
            if values
        },
        "paired_deltas": {
            "relational_mp_vs_joined_flat": _paired_delta_summary(
                task_records, candidate="relational_mp", reference="joined_flat"
            ),
            "relational_mp_vs_focal_only": _paired_delta_summary(
                task_records, candidate="relational_mp", reference="focal_only"
            ),
            "relational_mp_vs_relational_fk_shuffled": _paired_delta_summary(
                task_records, candidate="relational_mp", reference="relational_fk_shuffled"
            ),
            "relational_mp_vs_relational_parent_permuted": _paired_delta_summary(
                task_records, candidate="relational_mp", reference="relational_parent_permuted"
            ),
            "relational_mp_vs_relational_same_schema_no_fk": _paired_delta_summary(
                task_records, candidate="relational_mp", reference="relational_same_schema_no_fk"
            ),
        },
    }


def _paired_delta_summary(
    task_records: Sequence[dict[str, object]],
    candidate: str,
    reference: str,
    bootstrap_replicates: int = 2000,
) -> dict[str, float | int | None]:
    deltas: list[float] = []
    for record in task_records:
        scores = record.get("scores")
        if not isinstance(scores, dict):
            continue
        candidate_value = scores.get(candidate)
        reference_value = scores.get(reference)
        if isinstance(candidate_value, float) and isinstance(reference_value, float):
            deltas.append(candidate_value - reference_value)
    if not deltas:
        return {
            "count": 0,
            "mean": None,
            "bootstrap_ci95_low": None,
            "bootstrap_ci95_high": None,
            "win_rate": None,
        }
    values = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(0)
    indices = rng.integers(0, len(values), size=(bootstrap_replicates, len(values)))
    bootstrap_means = np.mean(values[indices], axis=1)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "bootstrap_ci95_low": float(np.percentile(bootstrap_means, 2.5)),
        "bootstrap_ci95_high": float(np.percentile(bootstrap_means, 97.5)),
        "win_rate": float(np.mean(values > 0.0)),
    }


def _attach_family_generation_stats(
    family_results: dict[str, dict[str, object]],
    sample_records: Sequence[dict[str, object]],
) -> None:
    records_by_family: dict[str, list[dict[str, object]]] = {}
    for record in sample_records:
        records_by_family.setdefault(str(record["family"]), []).append(record)

    for family, records in records_by_family.items():
        result = family_results.setdefault(family, {})
        attempts = [_int_value(record.get("attempts"), default=1) for record in records]
        database_retries = [
            _int_value(record.get("database_retries"), default=0) for record in records
        ]
        result["attempt_mean"] = float(np.mean(attempts)) if attempts else None
        result["attempt_p95"] = float(np.percentile(attempts, 95)) if attempts else None
        result["attempt_max"] = int(max(attempts)) if attempts else None
        result["database_retry_mean"] = (
            float(np.mean(database_retries)) if database_retries else None
        )
        result["database_retry_p95"] = (
            float(np.percentile(database_retries, 95)) if database_retries else None
        )
        result["database_retry_max"] = int(max(database_retries)) if database_retries else None
        checked_edges = sum(
            _int_value(record.get("checked_edges"), default=0) for record in records
        )
        edge_downgrades = sum(
            _int_value(record.get("edge_downgrade_count"), default=0) for record in records
        )
        result["edge_downgrade_rate"] = (
            None if checked_edges <= 0 else float(edge_downgrades) / float(checked_edges)
        )
        result["preflight_reason_histogram"] = _merge_histograms(
            record.get("preflight_reason_histogram") for record in records
        )
        result.update(_mandatory_fk_generation_rates(records))
        result["status"] = _family_status(family, result)


def _generation_health_summary(sample_records: Sequence[dict[str, object]]) -> dict[str, object]:
    attempts = [_int_value(record.get("attempts"), default=1) for record in sample_records]
    database_retries = [
        _int_value(record.get("database_retries"), default=0) for record in sample_records
    ]
    rates = _mandatory_fk_generation_rates(sample_records)
    retry_reason_histogram = {
        "balanced_task_resample": int(sum(max(attempt - 1, 0) for attempt in attempts)),
        "database_retry": int(sum(database_retries)),
        "edge_downgrade": int(
            sum(
                _int_value(record.get("edge_downgrade_count"), default=0)
                for record in sample_records
            )
        ),
        "edge_adjustment": int(
            sum(
                _int_value(record.get("edge_adjustment_count"), default=0)
                for record in sample_records
            )
        ),
        "mandatory_timestamp_resample": int(
            sum(
                _int_value(record.get("mandatory_fk_timestamp_resample_count"), default=0)
                for record in sample_records
            )
        ),
        "mandatory_backoff": int(
            sum(
                _int_value(record.get("mandatory_fk_backoff_count"), default=0)
                for record in sample_records
            )
        ),
        "mandatory_unsatisfied": int(
            sum(
                _int_value(record.get("mandatory_fk_unsatisfied"), default=0)
                for record in sample_records
            )
        ),
        "multi_parent_candidate_empty": int(
            sum(
                _int_value(record.get("multi_parent_candidate_empty_count"), default=0)
                for record in sample_records
            )
        ),
        "joint_sampler_backoff": int(
            sum(
                _int_value(record.get("joint_sampler_backoff_count"), default=0)
                for record in sample_records
            )
        ),
        "joint_sampler_independent_fallback": int(
            sum(
                _int_value(record.get("joint_sampler_independent_fallback_count"), default=0)
                for record in sample_records
            )
        ),
    }
    checked_edges = int(
        sum(_int_value(record.get("checked_edges"), default=0) for record in sample_records)
    )
    edge_downgrades = int(
        sum(_int_value(record.get("edge_downgrade_count"), default=0) for record in sample_records)
    )
    return {
        "sample_count": len(sample_records),
        "task_attempt_mean": float(np.mean(attempts)) if attempts else None,
        "task_attempt_p95": float(np.percentile(attempts, 95)) if attempts else None,
        "task_attempt_max": int(max(attempts)) if attempts else None,
        "database_retry_mean": float(np.mean(database_retries)) if database_retries else None,
        "database_retry_p95": float(np.percentile(database_retries, 95))
        if database_retries
        else None,
        "database_retry_max": int(max(database_retries)) if database_retries else None,
        "mandatory_fk_unsatisfied_after_repair": int(
            sum(
                _int_value(record.get("mandatory_fk_unsatisfied"), default=0)
                for record in sample_records
            )
        ),
        "mandatory_fk_forced_null": int(
            sum(
                _int_value(record.get("mandatory_fk_forced_null"), default=0)
                for record in sample_records
            )
        ),
        "natural_success_rate": rates["natural_success_rate"],
        "timestamp_resample_rate": rates["timestamp_resample_rate"],
        "constraint_backoff_rate": rates["constraint_backoff_rate"],
        "edge_downgrade_rate": None
        if checked_edges <= 0
        else float(edge_downgrades) / float(checked_edges),
        "preflight_reason_histogram": _merge_histograms(
            record.get("preflight_reason_histogram") for record in sample_records
        ),
        "preflight_action_histogram": _merge_histograms(
            record.get("preflight_action_histogram") for record in sample_records
        ),
        "task_rejection_histogram": _merge_histograms(
            record.get("task_rejection_histogram") for record in sample_records
        ),
        "retry_reason_histogram": retry_reason_histogram,
    }


def _mandatory_fk_generation_rates(
    records: Sequence[dict[str, object]],
) -> dict[str, float | int | None]:
    total = int(sum(_int_value(record.get("mandatory_fk_total"), default=0) for record in records))
    unsatisfied = int(
        sum(_int_value(record.get("mandatory_fk_unsatisfied"), default=0) for record in records)
    )
    backoff = int(
        sum(_int_value(record.get("mandatory_fk_backoff_count"), default=0) for record in records)
    )
    timestamp_resample = int(
        sum(
            _int_value(record.get("mandatory_fk_timestamp_resample_count"), default=0)
            for record in records
        )
    )
    if total <= 0:
        return {
            "mandatory_fk_total": total,
            "natural_success_rate": None,
            "timestamp_resample_rate": None,
            "constraint_backoff_rate": None,
        }
    repaired = min(total, unsatisfied + backoff + timestamp_resample)
    return {
        "mandatory_fk_total": total,
        "natural_success_rate": float(max(total - repaired, 0)) / float(total),
        "timestamp_resample_rate": float(timestamp_resample) / float(total),
        "constraint_backoff_rate": float(backoff) / float(total),
    }


def _schema_calibration_summary(schema_records: Sequence[dict[str, object]]) -> dict[str, object]:
    archetype_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    timestamp_counts: Counter[str] = Counter()
    total_tables = 0
    total_edges = 0
    total_temporal_edges = 0
    edge_density_values: list[float] = []
    structural_null_rate_values: list[float] = []
    dag_depth_values: list[int] = []
    root_ratio_values: list[float] = []
    bridge_present = 0
    snapshot_present = 0
    bridge_motif_violations = 0
    snapshot_motif_violations = 0
    constraint_totals: Counter[str] = Counter()
    for record in schema_records:
        archetype_counts[str(record.get("archetype", "unlabeled"))] += 1
        table_count = _int_value(record.get("table_count"), default=0)
        edge_count = _int_value(record.get("edge_count"), default=0)
        total_tables += table_count
        total_edges += edge_count
        total_temporal_edges += _int_value(record.get("temporal_fk_count"), default=0)
        role_counts.update(_counter_from_record(record.get("role_counts")))
        timestamp_counts.update(_counter_from_record(record.get("timestamp_counts_by_role")))
        edge_density_values.append(float(record.get("edge_density", 0.0)))
        structural_null_rate = record.get("structural_null_rate")
        if isinstance(structural_null_rate, float):
            structural_null_rate_values.append(structural_null_rate)
        dag_depth_values.append(_int_value(record.get("dag_depth"), default=0))
        root_ratio_values.append(float(record.get("root_table_ratio", 0.0)))
        bridge_present += int(bool(record.get("bridge_present", False)))
        snapshot_present += int(bool(record.get("snapshot_present", False)))
        bridge_motif_violations += int(bool(record.get("bridge_motif_violation", False)))
        snapshot_motif_violations += int(bool(record.get("snapshot_motif_violation", False)))
        constraint_totals.update(
            {
                key: _int_value(record.get(key), default=0)
                for key in (
                    "mandatory_fk_count",
                    "mandatory_null_violation_count",
                    "temporal_fk_count_for_validation",
                    "temporal_violation_count",
                    "one_to_one_fk_count",
                    "one_to_one_violation_count",
                    "capacity_fk_count",
                    "capacity_violation_count",
                )
            }
        )

    archetype_distribution = {
        archetype: float(count) / float(len(schema_records))
        for archetype, count in sorted(archetype_counts.items())
        if schema_records
    }
    role_distribution = {
        role: float(count) / float(total_tables)
        for role, count in sorted(role_counts.items())
        if total_tables > 0
    }
    timestamp_ratio_by_role = {
        role: float(timestamp_counts.get(role, 0)) / float(count)
        for role, count in sorted(role_counts.items())
        if count > 0
    }
    return {
        "sample_count": len(schema_records),
        "archetype_counts": dict(sorted(archetype_counts.items())),
        "archetype_distribution_realized": archetype_distribution,
        "role_counts": dict(sorted(role_counts.items())),
        "role_distribution_realized": role_distribution,
        "timestamp_table_ratio_by_role": timestamp_ratio_by_role,
        "temporal_fk_ratio": None
        if total_edges <= 0
        else float(total_temporal_edges) / float(total_edges),
        "bridge_realized_rate": None
        if not schema_records
        else float(bridge_present) / float(len(schema_records)),
        "snapshot_realized_rate": None
        if not schema_records
        else float(snapshot_present) / float(len(schema_records)),
        "bridge_motif_violation_count": bridge_motif_violations,
        "bridge_motif_violation_rate": (
            None
            if not schema_records
            else float(bridge_motif_violations) / float(len(schema_records))
        ),
        "snapshot_motif_violation_count": snapshot_motif_violations,
        "snapshot_motif_violation_rate": (
            None
            if not schema_records
            else float(snapshot_motif_violations) / float(len(schema_records))
        ),
        "structural_null_rate_mean": (
            float(np.mean(structural_null_rate_values)) if structural_null_rate_values else None
        ),
        "mandatory_null_violation_rate": _violation_rate(
            constraint_totals["mandatory_null_violation_count"],
            constraint_totals["mandatory_fk_count"],
        ),
        "temporal_violation_rate": _violation_rate(
            constraint_totals["temporal_violation_count"],
            constraint_totals["temporal_fk_count_for_validation"],
        ),
        "one_to_one_violation_rate": _violation_rate(
            constraint_totals["one_to_one_violation_count"],
            constraint_totals["one_to_one_fk_count"],
        ),
        "capacity_violation_rate": _violation_rate(
            constraint_totals["capacity_violation_count"], constraint_totals["capacity_fk_count"]
        ),
        "edge_density_mean": float(np.mean(edge_density_values)) if edge_density_values else None,
        "dag_depth_mean": float(np.mean(dag_depth_values)) if dag_depth_values else None,
        "root_table_ratio_mean": float(np.mean(root_ratio_values)) if root_ratio_values else None,
    }


def _benchmark_gate_summary(
    generation_health: dict[str, object] | None,
    schema_calibration: dict[str, object] | None,
    family_results: Mapping[str, Mapping[str, object]],
    suite: BenchmarkSuite = "base_balanced",
) -> dict[str, object]:
    failures: list[str] = []
    thresholds: dict[str, float] = {
        "database_retry_mean_max": 3.0,
        "database_retry_p95_max": 10.0,
        "task_attempt_p95_max": 50.0,
        "edge_downgrade_rate_max": 0.05,
        "natural_success_rate_min": 0.50,
        "bridge_motif_violation_rate_max": 0.0,
        "snapshot_motif_violation_rate_max": 0.0,
        "mandatory_null_violation_rate_max": 0.0,
        "temporal_violation_rate_max": 0.0,
        "one_to_one_violation_rate_max": 0.0,
        "capacity_violation_rate_max": 0.0,
    }
    if generation_health is not None:
        _check_max(
            failures,
            generation_health,
            "database_retry_mean",
            thresholds["database_retry_mean_max"],
        )
        _check_max(
            failures, generation_health, "database_retry_p95", thresholds["database_retry_p95_max"]
        )
        _check_max(
            failures, generation_health, "task_attempt_p95", thresholds["task_attempt_p95_max"]
        )
        _check_max(
            failures,
            generation_health,
            "edge_downgrade_rate",
            thresholds["edge_downgrade_rate_max"],
        )
        _check_min(
            failures,
            generation_health,
            "natural_success_rate",
            thresholds["natural_success_rate_min"],
        )
        if (
            _int_value(generation_health.get("mandatory_fk_unsatisfied_after_repair"), default=0)
            > 0
        ):
            failures.append("mandatory_fk_unsatisfied_after_repair > 0")
        if _int_value(generation_health.get("mandatory_fk_forced_null"), default=0) > 0:
            failures.append("mandatory_fk_forced_null > 0")
    if schema_calibration is not None:
        _check_max(
            failures,
            schema_calibration,
            "bridge_motif_violation_rate",
            thresholds["bridge_motif_violation_rate_max"],
        )
        _check_max(
            failures,
            schema_calibration,
            "snapshot_motif_violation_rate",
            thresholds["snapshot_motif_violation_rate_max"],
        )
        for metric in (
            "mandatory_null_violation_rate",
            "temporal_violation_rate",
            "one_to_one_violation_rate",
            "capacity_violation_rate",
        ):
            _check_max(failures, schema_calibration, metric, thresholds[f"{metric}_max"])
        if suite == "base_balanced":
            _check_max(failures, schema_calibration, "bridge_realized_rate", 0.0)
            _check_max(failures, schema_calibration, "snapshot_realized_rate", 0.0)
        elif suite == "many_to_many":
            _check_min(failures, schema_calibration, "bridge_realized_rate", 1.0)
        elif suite == "temporal_history":
            _check_min(failures, schema_calibration, "snapshot_realized_rate", 1.0)
    interaction = family_results.get("parent_child_interaction")
    if interaction is not None:
        gain_relational = interaction.get("gain_relational")
        if isinstance(gain_relational, float) and gain_relational < 0.0:
            failures.append("parent_child_interaction gain_relational < 0")
    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "thresholds": thresholds,
        "suite": suite,
    }


def _check_max(
    failures: list[str],
    metrics: Mapping[str, object],
    key: str,
    maximum: float,
) -> None:
    value = metrics.get(key)
    if isinstance(value, float) and value > maximum:
        failures.append(f"{key}={value:.6g} > {maximum:.6g}")


def _check_min(
    failures: list[str],
    metrics: Mapping[str, object],
    key: str,
    minimum: float,
) -> None:
    value = metrics.get(key)
    if isinstance(value, float) and value < minimum:
        failures.append(f"{key}={value:.6g} < {minimum:.6g}")


def _violation_rate(violation_count: int, eligible_count: int) -> float:
    return 0.0 if eligible_count <= 0 else float(violation_count) / float(eligible_count)


def _family_status(family: str, result: Mapping[str, object]) -> str:
    gain_relational = result.get("gain_relational")
    gain_joined = result.get("gain_joined")
    if not isinstance(gain_relational, float) or not isinstance(gain_joined, float):
        return "ok"
    if family == "parent_child_interaction" and gain_relational < 0.0:
        return "fail_interaction_or_message"
    if family == "parent_child_interaction" and gain_relational < gain_joined:
        return "relational_below_joined"
    return "ok"


def _root_table_count(database: RelationalDataset) -> int:
    children = {fk.child_table for fk in database.foreign_keys}
    return sum(1 for table_name in database.table_specs if table_name not in children)


def _dag_depth(database: RelationalDataset) -> int:
    children: dict[str, list[str]] = {table_name: [] for table_name in database.table_specs}
    indegree: dict[str, int] = {table_name: 0 for table_name in database.table_specs}
    for fk in database.foreign_keys:
        if fk.parent_table not in children or fk.child_table not in indegree:
            continue
        children[fk.parent_table].append(fk.child_table)
        indegree[fk.child_table] += 1
    queue = sorted(table_name for table_name, degree in indegree.items() if degree == 0)
    depth = {table_name: 1 for table_name in queue}
    while queue:
        table_name = queue.pop(0)
        for child in sorted(children[table_name]):
            depth[child] = max(depth.get(child, 1), depth[table_name] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return max(depth.values(), default=0)


def _counter_from_record(value: object) -> Counter[str]:
    if not isinstance(value, dict):
        return Counter()
    return Counter({str(key): _int_value(count, default=0) for key, count in value.items()})


def _merge_histograms(values: Iterable[object]) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for value in values:
        merged.update(_string_int_map(value))
    return dict(sorted(merged.items()))


def _string_int_map(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _int_value(count, default=0) for key, count in value.items()}


def _int_value(value: object, default: int) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and np.isfinite(value):
        return int(value)
    return default


def run_relational_benchmark(
    *,
    num_tasks: int = 10,
    seed: int = 0,
    config: RDBPriorConfig | None = None,
    ridge_alpha: float = 1.0,
) -> RelationalBenchmarkReport:
    generator = RelationalPriorGenerator(config or RDBPriorConfig(seed=seed))
    scores = _empty_score_map()
    score_task_types: dict[ProbeModel, list[TaskType]] = {name: [] for name in scores}
    family_scores: dict[str, dict[ProbeModel, list[float]]] = {}
    sample_records: list[dict[str, object]] = []
    schema_records: list[dict[str, object]] = []
    task_records: list[dict[str, object]] = []

    for task_idx in range(num_tasks):
        database = generator.sample_database()
        task = generator.sample_task(database)
        task_scores = _evaluate_task_scores(
            database=database,
            task=task,
            control_seed=seed + task_idx,
            ridge_alpha=ridge_alpha,
        )
        _append_task_scores(
            scores=scores,
            score_task_types=score_task_types,
            family_scores=family_scores,
            task=task,
            task_scores=task_scores,
        )
        sample_record = _benchmark_sample_record(
            database=database,
            task=task,
            sample_index=task_idx,
            seed=seed + task_idx,
            attempts=1,
        )
        sample_records.append(sample_record)
        task_records.append({**sample_record, "scores": dict(task_scores)})
        schema_records.append(_schema_record(database))

    return _summarize_relational_scores(
        scores=scores,
        score_task_types=score_task_types,
        family_scores=family_scores,
        sample_records=sample_records,
        schema_records=schema_records,
        task_records=task_records,
        evaluation_metadata={
            "probe_semantics": "ridge evaluation over constructed feature views; relational_mp is not a trained MP model",
            "task_filter_mode": "generator_config",
        },
    )


def run_balanced_relational_benchmark(
    *,
    seeds_per_cell: int = 50,
    seed: int = 0,
    config: RDBPriorConfig | None = None,
    ridge_alpha: float = 1.0,
    max_attempts_per_sample: int = 200,
    stability_splits: int = 5,
    target_families: tuple[RelationalTargetFamily, ...] = TARGET_FAMILIES,
    task_types: tuple[TaskType, ...] = TASK_TYPES,
    task_filter_mode: TaskFilterMode = "validity_only",
    suite: BenchmarkSuite = "base_balanced",
) -> RelationalBenchmarkReport:
    if seeds_per_cell < 1:
        raise ValueError("seeds_per_cell must be positive")
    if max_attempts_per_sample < 1:
        raise ValueError("max_attempts_per_sample must be positive")
    if stability_splits < 1:
        raise ValueError("stability_splits must be positive")
    if len(target_families) == 0:
        raise ValueError("target_families must be non-empty")
    if len(task_types) == 0:
        raise ValueError("task_types must be non-empty")
    if task_filter_mode not in ("validity_only", "signal_conditioned"):
        raise ValueError(f"unknown task_filter_mode {task_filter_mode!r}")
    if suite not in ("base_balanced", "many_to_many", "temporal_history"):
        raise ValueError(f"unknown benchmark suite {suite!r}")

    base_config = _benchmark_suite_config(
        config or RDBPriorConfig(seed=seed),
        task_filter_mode=task_filter_mode,
        suite=suite,
    )
    scores = _empty_score_map()
    score_task_types: dict[ProbeModel, list[TaskType]] = {name: [] for name in scores}
    family_scores: dict[str, dict[ProbeModel, list[float]]] = {}
    sample_records: list[dict[str, object]] = []
    sample_score_records: list[dict[str, object]] = []
    schema_records: list[dict[str, object]] = []

    for family_idx, family in enumerate(target_families):
        for task_type_idx, task_type in enumerate(task_types):
            for sample_idx in range(seeds_per_cell):
                sample_seed = _balanced_sample_seed(
                    seed=seed,
                    family_idx=family_idx,
                    task_type_idx=task_type_idx,
                    sample_idx=sample_idx,
                )
                database, task, accepted_seed, attempts, rejection_histogram = (
                    _sample_balanced_task(
                        base_config=base_config,
                        family=family,
                        task_type=task_type,
                        sample_seed=sample_seed,
                        max_attempts_per_sample=max_attempts_per_sample,
                    )
                )
                task_scores = _evaluate_task_scores(
                    database=database,
                    task=task,
                    control_seed=accepted_seed,
                    ridge_alpha=ridge_alpha,
                )
                _append_task_scores(
                    scores=scores,
                    score_task_types=score_task_types,
                    family_scores=family_scores,
                    task=task,
                    task_scores=task_scores,
                )
                sample_record = _benchmark_sample_record(
                    database=database,
                    task=task,
                    sample_index=sample_idx,
                    seed=accepted_seed,
                    attempts=attempts,
                    task_rejection_histogram=rejection_histogram,
                    requested_family=family,
                    requested_task_type=task_type,
                )
                sample_records.append(sample_record)
                sample_score_records.append({**sample_record, "scores": dict(task_scores)})
                schema_records.append(_schema_record(database))

    sampling_summary = _balanced_sampling_summary(
        samples=sample_records,
        target_families=target_families,
        task_types=task_types,
        seeds_per_cell=seeds_per_cell,
        seed=seed,
        max_attempts_per_sample=max_attempts_per_sample,
    )
    stability_results = _balanced_stability_results(
        samples=sample_score_records,
        target_families=target_families,
        task_types=task_types,
        seeds_per_cell=seeds_per_cell,
        stability_splits=stability_splits,
    )
    return _summarize_relational_scores(
        scores=scores,
        score_task_types=score_task_types,
        family_scores=family_scores,
        sampling_summary=sampling_summary,
        stability_results=stability_results,
        sample_records=sample_records,
        schema_records=schema_records,
        task_records=sample_score_records,
        evaluation_metadata={
            "probe_semantics": "ridge evaluation over constructed feature views; relational_mp is an enriched view, not a trained MP model",
            "control_semantics": "joined_* corrupt joined features; relational_* corrupt the complete enriched relational view",
            "task_filter_mode": task_filter_mode,
            "suite": suite,
        },
        suite=suite,
    )


def _balanced_sample_seed(
    seed: int,
    family_idx: int,
    task_type_idx: int,
    sample_idx: int,
) -> int:
    return seed + family_idx * 1_000_000 + task_type_idx * 100_000 + sample_idx * 1_000


def _sample_balanced_task(
    base_config: RDBPriorConfig,
    family: RelationalTargetFamily,
    task_type: TaskType,
    sample_seed: int,
    max_attempts_per_sample: int,
) -> tuple[RelationalDataset, RelationalTask, int, int, dict[str, int]]:
    last_error: Exception | None = None
    rejection_histogram: Counter[str] = Counter()
    for attempt in range(max_attempts_per_sample):
        attempt_seed = sample_seed + attempt
        config = replace(
            base_config,
            seed=attempt_seed,
            task=task_type,
            explicit_relational_target_probability=1.0,
        )
        generator = RelationalPriorGenerator(config)
        try:
            database = generator.sample_database()
            task = generator.sample_explicit_task(database, family=family, task_type=task_type)
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            rejection_histogram["database_or_task_error"] += 1
            continue
        if task is None:
            rejection_histogram[generator.last_task_rejection_reason or "task_unavailable"] += 1
            continue
        if task.target_spec is None or task.target_spec.target_family != family:
            continue
        if task.task_type != task_type:
            continue
        return database, task, attempt_seed, attempt + 1, dict(rejection_histogram)
    message = (
        f"could not sample balanced RDB task for family={family!r}, task_type={task_type!r} "
        f"after {max_attempts_per_sample} attempts"
    )
    if last_error is not None:
        raise RuntimeError(message) from last_error
    raise RuntimeError(message)


def _benchmark_suite_config(
    config: RDBPriorConfig,
    task_filter_mode: TaskFilterMode,
    suite: BenchmarkSuite,
) -> RDBPriorConfig:
    configured = replace(
        config,
        explicit_target_difficulty_filter=task_filter_mode == "signal_conditioned",
    )
    if suite == "many_to_many":
        return replace(
            configured,
            enable_many_to_many_motif=True,
            schema_archetype=SchemaArchetypeConfig(forced_archetype="many-to-many"),
        )
    if suite == "temporal_history":
        return replace(
            configured,
            enable_snapshot_tables=True,
            schema_archetype=SchemaArchetypeConfig(forced_archetype="temporal-history"),
        )
    return configured


def _balanced_sampling_summary(
    samples: list[dict[str, object]],
    target_families: tuple[RelationalTargetFamily, ...],
    task_types: tuple[TaskType, ...],
    seeds_per_cell: int,
    seed: int,
    max_attempts_per_sample: int,
) -> dict[str, object]:
    cell_counts = Counter(
        _cell_key(str(sample["family"]), str(sample["task_type"])) for sample in samples
    )
    family_counts = Counter(str(sample["family"]) for sample in samples)
    task_type_counts = Counter(str(sample["task_type"]) for sample in samples)
    expected_cells = [
        _cell_key(family, task_type) for family in target_families for task_type in task_types
    ]
    missing_or_short = {
        cell: int(cell_counts.get(cell, 0))
        for cell in expected_cells
        if int(cell_counts.get(cell, 0)) < seeds_per_cell
    }
    attempts = [int(sample["attempts"]) for sample in samples]
    generation_health = _generation_health_summary(samples)
    return {
        "sampling_mode": "family_task_type_balanced",
        "seed": seed,
        "seed_formula": "seed + family_idx*1000000 + task_type_idx*100000 + sample_idx*1000 + retry_attempt",
        "seeds_per_cell": seeds_per_cell,
        "target_families": list(target_families),
        "task_types": list(task_types),
        "expected_cell_count": len(expected_cells),
        "sample_count": len(samples),
        "cell_counts": dict(sorted(cell_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "task_type_counts": dict(sorted(task_type_counts.items())),
        "passed_min_cell_count": len(missing_or_short) == 0,
        "missing_or_short_cells": missing_or_short,
        "max_attempts_per_sample": max_attempts_per_sample,
        "attempt_mean": float(np.mean(attempts)) if attempts else None,
        "attempt_p95": float(np.percentile(attempts, 95)) if attempts else None,
        "attempt_max": int(max(attempts)) if attempts else None,
        "task_attempt_mean": generation_health["task_attempt_mean"],
        "task_attempt_p95": generation_health["task_attempt_p95"],
        "database_retry_mean": generation_health["database_retry_mean"],
        "database_retry_p95": generation_health["database_retry_p95"],
        "natural_success_rate": generation_health["natural_success_rate"],
        "timestamp_resample_rate": generation_health["timestamp_resample_rate"],
        "constraint_backoff_rate": generation_health["constraint_backoff_rate"],
        "retry_reason_histogram": generation_health["retry_reason_histogram"],
        "samples": samples,
    }


def _balanced_stability_results(
    samples: list[dict[str, object]],
    target_families: tuple[RelationalTargetFamily, ...],
    task_types: tuple[TaskType, ...],
    seeds_per_cell: int,
    stability_splits: int,
) -> dict[str, object]:
    distribution = _balanced_distribution_stability(
        samples=samples,
        target_families=target_families,
        task_types=task_types,
        seeds_per_cell=seeds_per_cell,
    )
    split_summaries: list[dict[str, object]] = []
    for split_idx in range(stability_splits):
        split_samples = [
            sample
            for sample in samples
            if int(sample["sample_index"]) % stability_splits == split_idx
        ]
        split_summaries.append(_split_score_summary(split_idx, split_samples))
    return {
        "distribution": distribution,
        "score_splits": split_summaries,
        "score_stability": _gain_stability_summary(split_summaries),
    }


def _balanced_distribution_stability(
    samples: list[dict[str, object]],
    target_families: tuple[RelationalTargetFamily, ...],
    task_types: tuple[TaskType, ...],
    seeds_per_cell: int,
) -> dict[str, object]:
    cell_counts = Counter(
        _cell_key(str(sample["family"]), str(sample["task_type"])) for sample in samples
    )
    expected_cells = [
        _cell_key(family, task_type) for family in target_families for task_type in task_types
    ]
    deviations = {cell: int(cell_counts.get(cell, 0)) - seeds_per_cell for cell in expected_cells}
    max_abs_deviation = max((abs(value) for value in deviations.values()), default=0)
    return {
        "passed": max_abs_deviation == 0,
        "expected_per_cell": seeds_per_cell,
        "cell_counts": {cell: int(cell_counts.get(cell, 0)) for cell in expected_cells},
        "cell_count_deviation": deviations,
        "max_abs_cell_count_deviation": max_abs_deviation,
    }


def _split_score_summary(split_idx: int, samples: list[dict[str, object]]) -> dict[str, object]:
    model_scores: dict[str, list[float]] = {name: [] for name in _empty_score_map()}
    for sample in samples:
        score_map = sample["scores"]
        if not isinstance(score_map, dict):
            continue
        for model_name in model_scores:
            value = score_map.get(model_name)
            if isinstance(value, float):
                model_scores[model_name].append(value)
    focal = _mean_or_none(model_scores["focal_only"])
    joined = _mean_or_none(model_scores["joined_flat"])
    relational = _mean_or_none(model_scores["relational_mp"])
    independent = _mean_or_none(model_scores["independent_single_table"])
    return {
        "split_index": split_idx,
        "sample_count": len(samples),
        "cell_counts": dict(
            sorted(
                Counter(_cell_key(str(s["family"]), str(s["task_type"])) for s in samples).items()
            )
        ),
        "gain_joined": None if focal is None or joined is None else joined - focal,
        "gain_relational": None if focal is None or relational is None else relational - focal,
        "gain_vs_independent": None
        if relational is None or independent is None
        else relational - independent,
    }


def _gain_stability_summary(split_summaries: list[dict[str, object]]) -> dict[str, object]:
    return {
        "gain_joined": _numeric_summary(
            [
                summary["gain_joined"]
                for summary in split_summaries
                if summary["gain_joined"] is not None
            ]
        ),
        "gain_relational": _numeric_summary(
            [
                summary["gain_relational"]
                for summary in split_summaries
                if summary["gain_relational"] is not None
            ]
        ),
        "gain_vs_independent": _numeric_summary(
            [
                summary["gain_vs_independent"]
                for summary in split_summaries
                if summary["gain_vs_independent"] is not None
            ]
        ),
    }


def _numeric_summary(values: list[object]) -> dict[str, float | int | None]:
    numeric = np.asarray([float(value) for value in values], dtype=np.float64)
    if numeric.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(numeric.size),
        "mean": float(np.mean(numeric)),
        "std": float(np.std(numeric)),
        "min": float(np.min(numeric)),
        "max": float(np.max(numeric)),
    }


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _cell_key(family: str, task_type: str) -> str:
    return f"{family}::{task_type}"


def run_ablation_suite(seed: int = 0, num_tasks: int = 5) -> dict[str, object]:
    ablations: dict[str, object] = {}
    configs: Mapping[str, RDBPriorConfig] = {
        "default_trap": RDBPriorConfig(seed=seed),
        "uniform_attachment": RDBPriorConfig(
            seed=seed,
            mechanism_hyperprior=MechanismHyperpriorConfig(forced_attachment="uniform"),
        ),
        "parent_latent_only": RDBPriorConfig(seed=seed, use_parent_feature_messages=False),
        "independent_fk": RDBPriorConfig(seed=seed, multi_parent_probability=0.0),
    }
    for name, config in configs.items():
        ablations[name] = run_relational_benchmark(
            num_tasks=num_tasks, seed=seed, config=config
        ).to_dict()
    return ablations
