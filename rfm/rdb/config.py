from __future__ import annotations

from dataclasses import dataclass, field

from rfm.config import PriorConfig
from rfm.rdb.types import (
    AttachmentMode,
    ExistenceMode,
    FeatureColumnType,
    MandatoryFKPolicy,
    RelationProfileKind,
    SchemaArchetype,
    TableRole,
)
from rfm.types import TaskType


@dataclass(frozen=True)
class RoleGrammarConfig:
    timestamp_probability_by_role: dict[TableRole, float] = field(
        default_factory=lambda: {
            "entity": 0.20,
            "activity/event": 0.85,
            "dimension/lookup": 0.02,
            "bridge": 0.30,
            "snapshot/state": 0.95,
        }
    )


@dataclass(frozen=True)
class SchemaArchetypeConfig:
    distribution: tuple[tuple[SchemaArchetype, float], ...] = (
        ("star", 0.25),
        ("snowflake", 0.25),
        ("entity-event", 0.25),
        ("event-lookup", 0.25),
        ("many-to-many", 0.15),
        ("temporal-history", 0.10),
    )
    forced_archetype: SchemaArchetype | None = None


@dataclass(frozen=True)
class MechanismHyperpriorConfig:
    field_weight_scale: float = 1.0
    temperature_min: float = 0.3
    temperature_max: float = 2.5
    existence_bias_range: tuple[float, float] = (-2.0, 2.0)
    existence_latent_scale: float = 0.8
    existence_time_scale: float = 0.5
    hub_strength_range: tuple[float, float] = (0.5, 2.5)
    locality_strength_range: tuple[float, float] = (0.5, 2.5)
    temporal_strength_range: tuple[float, float] = (0.5, 2.5)
    compat_strength_range: tuple[float, float] = (0.3, 1.5)
    sparse_existence_bias: float = 0.8
    optional_existence_bias: float = 1.4
    mandatory_existence_bias: float = 2.5
    forced_attachment: AttachmentMode | None = None
    forced_existence: ExistenceMode | None = None


@dataclass(frozen=True)
class MultiParentConfig:
    max_joint_parents: int = 3
    candidate_trim: int = 12
    latent_compat_weight: float = 2.0
    rank_compat_weight: float = 0.5
    segment_compat_weight: float = 2.0
    gibbs_iterations: int = 8
    bridge_pair_energy_weight: float = 1.5


@dataclass(frozen=True)
class RelationalSCMConfig:
    prior: PriorConfig = field(default_factory=PriorConfig)
    min_hidden_nodes: int = 4
    max_hidden_nodes: int = 24
    feature_missing_probability: float = 0.20
    max_feature_missing_rate: float = 0.20


@dataclass(frozen=True)
class RDBPriorConfig:
    min_tables: int = 4
    max_tables: int = 6
    min_rows_per_table: int = 24
    max_rows_per_table: int = 160
    min_features_per_table: int = 2
    max_features_per_table: int = 8
    latent_dim: int = 8
    max_foreign_keys_per_table: int = 3
    relation_profiles: tuple[RelationProfileKind, ...] = (
        "uniform",
        "popularity",
        "locality",
        "temporal",
        "capacity",
        "multi_parent",
        "hybrid",
    )
    feature_types: tuple[FeatureColumnType, ...] = (
        "continuous",
        "categorical",
        "ordinal",
        "binary",
        "count",
        "quantized",
        "high_cardinality_categorical",
    )
    table_roles: tuple[TableRole, ...] = (
        "entity",
        "activity/event",
        "bridge",
        "dimension/lookup",
        "snapshot/state",
    )
    optional_foreign_key_probability: float = 0.12
    one_to_one_probability: float = 0.25
    capacity_limited_probability: float = 0.45
    temporal_foreign_key_probability: float = 0.35
    multi_parent_probability: float = 0.35
    enable_many_to_many_motif: bool = False
    enable_snapshot_tables: bool = False
    feature_missing_probability: float = 0.20
    max_feature_missing_rate: float = 0.20
    temporal_task_probability: float = 0.35
    ood_task_probability: float = 0.15
    task: TaskType | None = None
    seed: int | None = None
    role_grammar: RoleGrammarConfig = field(default_factory=RoleGrammarConfig)
    schema_archetype: SchemaArchetypeConfig = field(default_factory=SchemaArchetypeConfig)
    mechanism_hyperprior: MechanismHyperpriorConfig = field(
        default_factory=MechanismHyperpriorConfig
    )
    multi_parent: MultiParentConfig = field(default_factory=MultiParentConfig)
    relational_scm: RelationalSCMConfig = field(default_factory=RelationalSCMConfig)
    topology_context_dim: int = 6
    mandatory_fk_policy: MandatoryFKPolicy = "resample_child_time"
    mandatory_fk_max_retries: int = 8
    cross_table_target_probability: float = 0.30
    explicit_relational_target_probability: float = 0.70
    explicit_target_difficulty_filter: bool = True
    explicit_target_min_class_entropy: float = 0.65
    explicit_target_min_class_fraction: float = 0.08
    explicit_target_min_classification_probe_margin: float = 0.03
    explicit_target_max_classification_probe_accuracy: float = 0.98
    explicit_target_min_regression_probe_r2: float = 0.02
    explicit_target_max_regression_probe_r2: float = 0.98
    explicit_target_min_relational_probe_gain: float = 0.005
    parent_feature_message_dim: int = 3
    use_parent_feature_messages: bool = True

    def __post_init__(self) -> None:
        if self.min_tables < 4 or self.min_tables > self.max_tables:
            raise ValueError("archetype schema range must satisfy 4 <= min_tables <= max_tables")
        if self.min_rows_per_table < 2 or self.min_rows_per_table > self.max_rows_per_table:
            raise ValueError("row range must satisfy 2 <= min_rows_per_table <= max_rows_per_table")
        if (
            self.min_features_per_table < 1
            or self.min_features_per_table > self.max_features_per_table
        ):
            raise ValueError(
                "feature range must satisfy 1 <= min_features_per_table <= max_features_per_table"
            )
        if self.latent_dim < 2:
            raise ValueError("latent_dim must be at least 2")
        if self.max_foreign_keys_per_table < 1:
            raise ValueError("max_foreign_keys_per_table must be positive")
        if not 0.0 <= self.optional_foreign_key_probability <= 1.0:
            raise ValueError("optional_foreign_key_probability must be in [0, 1]")
        if not 0.0 <= self.one_to_one_probability <= 1.0:
            raise ValueError("one_to_one_probability must be in [0, 1]")
        if not 0.0 <= self.capacity_limited_probability <= 1.0:
            raise ValueError("capacity_limited_probability must be in [0, 1]")
        if not 0.0 <= self.temporal_foreign_key_probability <= 1.0:
            raise ValueError("temporal_foreign_key_probability must be in [0, 1]")
        if not 0.0 <= self.multi_parent_probability <= 1.0:
            raise ValueError("multi_parent_probability must be in [0, 1]")
        if not 0.0 <= self.feature_missing_probability <= 1.0:
            raise ValueError("feature_missing_probability must be in [0, 1]")
        if not 0.0 <= self.max_feature_missing_rate <= 1.0:
            raise ValueError("max_feature_missing_rate must be in [0, 1]")
        if not 0.0 <= self.temporal_task_probability <= 1.0:
            raise ValueError("temporal_task_probability must be in [0, 1]")
        if not 0.0 <= self.ood_task_probability <= 1.0:
            raise ValueError("ood_task_probability must be in [0, 1]")
        if len(self.relation_profiles) == 0:
            raise ValueError("relation_profiles must be non-empty")
        if len(self.feature_types) == 0:
            raise ValueError("feature_types must be non-empty")
        if len(self.table_roles) == 0:
            raise ValueError("table_roles must be non-empty")
        for role, probability in self.role_grammar.timestamp_probability_by_role.items():
            if role not in self.table_roles:
                continue
            if not 0.0 <= probability <= 1.0:
                raise ValueError("timestamp_probability_by_role probabilities must be in [0, 1]")
        if len(self.schema_archetype.distribution) == 0:
            raise ValueError("schema_archetype.distribution must be non-empty")
        for _, weight in self.schema_archetype.distribution:
            if weight < 0.0:
                raise ValueError("schema archetype weights must be non-negative")
        if (
            self.schema_archetype.forced_archetype == "many-to-many"
            and not self.enable_many_to_many_motif
        ):
            raise ValueError("many-to-many archetype requires enable_many_to_many_motif=True")
        if (
            self.schema_archetype.forced_archetype == "temporal-history"
            and not self.enable_snapshot_tables
        ):
            raise ValueError("temporal-history archetype requires enable_snapshot_tables=True")
        if self.topology_context_dim < 1:
            raise ValueError("topology_context_dim must be positive")
        if self.mandatory_fk_max_retries < 1:
            raise ValueError("mandatory_fk_max_retries must be positive")
        if not 0.0 <= self.cross_table_target_probability <= 1.0:
            raise ValueError("cross_table_target_probability must be in [0, 1]")
        if not 0.0 <= self.explicit_relational_target_probability <= 1.0:
            raise ValueError("explicit_relational_target_probability must be in [0, 1]")
        if not 0.0 <= self.explicit_target_min_class_entropy <= 1.0:
            raise ValueError("explicit_target_min_class_entropy must be in [0, 1]")
        if not 0.0 <= self.explicit_target_min_class_fraction <= 1.0:
            raise ValueError("explicit_target_min_class_fraction must be in [0, 1]")
        if not 0.0 <= self.explicit_target_min_classification_probe_margin <= 1.0:
            raise ValueError("explicit_target_min_classification_probe_margin must be in [0, 1]")
        if not 0.0 <= self.explicit_target_max_classification_probe_accuracy <= 1.0:
            raise ValueError("explicit_target_max_classification_probe_accuracy must be in [0, 1]")
        if (
            self.explicit_target_min_regression_probe_r2
            > self.explicit_target_max_regression_probe_r2
        ):
            raise ValueError(
                "explicit_target_min_regression_probe_r2 must be <= explicit_target_max_regression_probe_r2"
            )
        if self.explicit_target_min_relational_probe_gain < 0.0:
            raise ValueError("explicit_target_min_relational_probe_gain must be non-negative")
        if self.parent_feature_message_dim < 0:
            raise ValueError("parent_feature_message_dim must be non-negative")
