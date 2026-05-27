from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from rfm.types import TaskType

TableRole = Literal["entity", "activity/event", "bridge", "dimension/lookup", "snapshot/state"]
SchemaArchetype = Literal[
    "star", "snowflake", "entity-event", "event-lookup", "many-to-many", "temporal-history"
]
ColumnKind = Literal["primary_key", "timestamp", "foreign_key", "feature"]
FeatureColumnType = Literal[
    "continuous",
    "categorical",
    "ordinal",
    "binary",
    "count",
    "quantized",
    "high_cardinality_categorical",
]
RelationProfileKind = Literal[
    "uniform",
    "popularity",
    "locality",
    "temporal",
    "capacity",
    "multi_parent",
    "hybrid",
]
ForeignKeyCardinality = Literal[
    "many_to_one", "one_to_one", "capacity_limited", "optional", "multi_parent_member"
]
RelationalSplitKind = Literal["random", "temporal", "ood"]
MandatoryFKPolicy = Literal["resample_child_time", "backoff_parent_pool", "hard_fail_resample_edge"]
RelationalTargetFamily = Literal[
    "local_only",
    "parent_feature",
    "parent_child_interaction",
    "multi_parent",
    "topology_driven",
]
TargetDependencyKind = Literal[
    "focal_only",
    "parent_feature",
    "joined",
    "parent_child_interaction",
    "multi_parent",
    "topology_driven",
]
TargetDifficultyProbeView = Literal["focal_only", "joined_flat", "topology"]

EdgeSemantic = Literal[
    "entity_belongs_to_lookup",
    "activity_refs_entity",
    "activity_refs_dimension",
    "bridge_pairs_entities",
    "snapshot_refs_entity_or_activity",
]
EdgeIntent = EdgeSemantic

ExistenceMode = Literal["mandatory", "optional", "sparse"]
AttachmentMode = Literal[
    "uniform", "hub_preferential", "segment_local", "temporal_causal", "bridge_pairing"
]
CoordinationMode = Literal["independent", "joint_tuple", "bridge_pair"]
CapacityMode = Literal["unbounded", "one_to_one", "k_limited"]


@dataclass(frozen=True)
class EdgeIntentSpec:
    intent: EdgeIntent
    allowed_cardinalities: tuple[ForeignKeyCardinality, ...]
    default_temporal: bool
    coordination: CoordinationMode


@dataclass(frozen=True)
class MechanismProfile:
    existence: ExistenceMode
    attachment: AttachmentMode
    coordination: Literal["independent", "joint_tuple"]
    field_weights: tuple[float, ...]
    temperature: float
    capacity_mode: CapacityMode
    existence_bias: float = 0.0
    existence_latent_weight: tuple[float, ...] | None = None
    existence_time_weight: float = 0.0
    hub_strength: float = 0.0
    locality_strength: float = 0.0
    temporal_strength: float = 0.0
    bridge_same_segment_bias: float = 0.0
    compat_strength: float = 1.0
    noise_scale: float = 0.1
    capacity_k: int | None = None

    @property
    def field_weights_array(self) -> NDArray[np.float64]:
        return np.asarray(self.field_weights, dtype=np.float64)

    @property
    def existence_latent_weight_array(self) -> NDArray[np.float64] | None:
        if self.existence_latent_weight is None:
            return None
        return np.asarray(self.existence_latent_weight, dtype=np.float64)


@dataclass(frozen=True)
class RelationProfile:
    """Legacy alias kept for metadata compatibility; prefer MechanismProfile."""

    kind: RelationProfileKind
    nullable_rate: float = 0.0
    popularity_strength: float = 0.0
    locality_strength: float = 0.0
    temporal_strength: float = 0.0
    compatibility_strength: float = 1.0
    capacity_strength: float = 0.0
    multi_parent_strength: float = 0.0
    noise_scale: float = 0.1


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: ColumnKind
    value_type: FeatureColumnType | Literal["integer", "timestamp"] = "continuous"
    nullable: bool = False
    source: str | None = None


@dataclass(frozen=True)
class TableSpec:
    name: str
    role: TableRole
    row_count: int
    columns: tuple[ColumnSpec, ...]
    primary_key: str
    timestamp_column: str | None
    has_timestamp: bool = True

    def __post_init__(self) -> None:
        if self.has_timestamp != (self.timestamp_column is not None):
            raise ValueError("has_timestamp must match whether timestamp_column is set")

    @property
    def feature_columns(self) -> tuple[ColumnSpec, ...]:
        return tuple(column for column in self.columns if column.kind == "feature")

    @property
    def foreign_key_columns(self) -> tuple[ColumnSpec, ...]:
        return tuple(column for column in self.columns if column.kind == "foreign_key")

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def column_index(self, column_name: str) -> int:
        for idx, column in enumerate(self.columns):
            if column.name == column_name:
                return idx
        raise KeyError(f"unknown column {column_name!r} in table {self.name!r}")


@dataclass(frozen=True)
class ForeignKeySpec:
    child_table: str
    child_column: str
    parent_table: str
    parent_column: str
    cardinality: ForeignKeyCardinality
    nullable: bool
    capacity: int | None
    temporal: bool
    mechanism: MechanismProfile
    intent: EdgeIntent
    multi_parent_group: str | None = None
    semantic: EdgeSemantic | None = None
    existence: ExistenceMode | None = None

    def __post_init__(self) -> None:
        if self.semantic is None:
            object.__setattr__(self, "semantic", self.intent)
        if self.existence is None:
            object.__setattr__(self, "existence", self.mechanism.existence)
        if self.intent != self.semantic:
            raise ValueError("intent compatibility field must match semantic")
        if self.existence != self.mechanism.existence:
            raise ValueError("foreign key existence must match mechanism existence")
        if self.existence == "mandatory" and self.nullable:
            raise ValueError("mandatory foreign keys cannot be nullable")

    @property
    def key(self) -> str:
        return f"{self.child_table}.{self.child_column}"


@dataclass(frozen=True)
class SchemaGraph:
    tables: dict[str, TableSpec]
    edges: tuple[ForeignKeySpec, ...]
    edge_intents: dict[str, EdgeIntentSpec]
    topological_order: tuple[str, ...]
    archetype: SchemaArchetype | None = None


@dataclass(frozen=True)
class RelationalDataset:
    tables: dict[str, NDArray[np.float32]]
    table_specs: dict[str, TableSpec]
    foreign_keys: tuple[ForeignKeySpec, ...]
    foreign_key_null_masks: dict[str, NDArray[np.bool_]]
    feature_missing_masks: dict[str, NDArray[np.bool_]]
    row_embeddings: dict[str, NDArray[np.float32]]
    metadata: dict[str, object]

    def column_index(self, table_name: str, column_name: str) -> int:
        return self.table_specs[table_name].column_index(column_name)

    def column_values(self, table_name: str, column_name: str) -> NDArray[np.float32]:
        table = self.tables[table_name]
        return table[:, self.column_index(table_name, column_name)]


@dataclass(frozen=True)
class MandatoryFKStats:
    mandatory_fk_total: int
    mandatory_fk_unsatisfied: int
    mandatory_fk_forced_null: int
    mandatory_fk_backoff_count: int
    mandatory_fk_timestamp_resample_count: int = 0
    multi_parent_candidate_empty_count: int = 0
    joint_sampler_backoff_count: int = 0
    joint_sampler_independent_fallback_count: int = 0

    @property
    def mandatory_fk_unsatisfied_rate(self) -> float:
        if self.mandatory_fk_total <= 0:
            return 0.0
        return float(self.mandatory_fk_unsatisfied) / float(self.mandatory_fk_total)

    @property
    def timestamp_resample_rate(self) -> float:
        if self.mandatory_fk_total <= 0:
            return 0.0
        return float(self.mandatory_fk_timestamp_resample_count) / float(self.mandatory_fk_total)

    @property
    def constraint_backoff_rate(self) -> float:
        if self.mandatory_fk_total <= 0:
            return 0.0
        return float(self.mandatory_fk_backoff_count) / float(self.mandatory_fk_total)

    @property
    def natural_mandatory_success_rate(self) -> float:
        if self.mandatory_fk_total <= 0:
            return 1.0
        repaired = (
            self.mandatory_fk_unsatisfied
            + self.mandatory_fk_backoff_count
            + self.mandatory_fk_timestamp_resample_count
        )
        natural = max(self.mandatory_fk_total - repaired, 0)
        return float(natural) / float(self.mandatory_fk_total)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mandatory_fk_total": self.mandatory_fk_total,
            "mandatory_fk_unsatisfied": self.mandatory_fk_unsatisfied,
            "mandatory_fk_forced_null": self.mandatory_fk_forced_null,
            "mandatory_fk_backoff_count": self.mandatory_fk_backoff_count,
            "mandatory_fk_timestamp_resample_count": self.mandatory_fk_timestamp_resample_count,
            "multi_parent_candidate_empty_count": self.multi_parent_candidate_empty_count,
            "joint_sampler_backoff_count": self.joint_sampler_backoff_count,
            "joint_sampler_independent_fallback_count": self.joint_sampler_independent_fallback_count,
            "mandatory_fk_unsatisfied_rate": self.mandatory_fk_unsatisfied_rate,
            "timestamp_resample_rate": self.timestamp_resample_rate,
            "constraint_backoff_rate": self.constraint_backoff_rate,
            "natural_mandatory_success_rate": self.natural_mandatory_success_rate,
        }


@dataclass(frozen=True)
class ParentMessage:
    fk_key: str
    parent_table: str
    parent_latent: NDArray[np.float64]
    parent_features: NDArray[np.float64]
    edge_profile: NDArray[np.float64]
    fanout_state: float
    is_null: bool


@dataclass(frozen=True)
class RelationMessageSet:
    messages: tuple[ParentMessage, ...]
    edge_messages: NDArray[np.float64]


@dataclass(frozen=True)
class TargetDifficultyMetrics:
    accepted_view: TargetDifficultyProbeView
    accepted_score: float
    focal_score: float
    joined_score: float
    topology_score: float
    baseline_score: float | None = None
    class_entropy: float | None = None
    min_class_fraction: float | None = None
    relational_gain: float | None = None


@dataclass(frozen=True)
class RelationalTargetSpec:
    focal_table: str
    target_column: str
    target_family: RelationalTargetFamily
    join_path: tuple[str, ...]
    parent_tables_used: tuple[str, ...]
    parent_columns_used: tuple[str, ...]
    topology_features_used: tuple[str, ...]
    causal_formula_type: str
    signal_to_noise_ratio: float
    focal_only_expected_difficulty: float
    difficulty_metrics: TargetDifficultyMetrics | None = None


@dataclass(frozen=True)
class RelationalTask:
    target_table: str
    target_column: str
    task_type: TaskType
    train_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    split_kind: RelationalSplitKind
    num_classes: int | None = None
    has_cross_table_path: bool = False
    parent_tables_in_path: tuple[str, ...] = ()
    target_dependency_kind: TargetDependencyKind = "focal_only"
    causal_feature_sources: dict[str, tuple[str, ...]] | None = None
    target_spec: RelationalTargetSpec | None = None
    target_values: NDArray[np.float32] | None = None


@dataclass(frozen=True)
class RelationalPretrainSpec:
    """Skeleton contract for future relational foundation models."""

    database: RelationalDataset
    task: RelationalTask
    focal_table: str
    neighbor_tables: tuple[str, ...]
    join_paths: tuple[tuple[str, ...], ...]
    topology_summary: dict[str, object]
