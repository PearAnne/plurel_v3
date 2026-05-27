from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TaskMode = Literal["classification", "regression", "mixed"]
PriorMode = Literal["scm", "tree", "gp", "bag"]
GraphLayout = Literal[
    "layered",
    "erdos_renyi",
    "barabasi_albert",
    "random_tree",
    "reverse_tree",
    "random_cauchy",
    "block",
    "hub",
    "small_world",
]
AggregationKind = Literal[
    "sum",
    "mean",
    "max",
    "product",
    "logexp",
    "gated_sum",
    "attention",
    "quadratic",
]
ActivationKind = Literal[
    "identity",
    "relu",
    "tanh",
    "sigmoid",
    "gelu",
    "softplus",
    "abs",
    "log_signed",
    "sine",
    "cosine",
]
SourceKind = Literal[
    "normal",
    "uniform",
    "beta",
    "zipf",
    "mixed",
    "trend",
    "cycle",
    "multi_sine",
    "ar1",
    "seasonal_ar",
    "event",
]
EdgeWeightKind = Literal["gaussian", "uniform", "lognormal", "cauchy"]
EdgeMechanismKind = Literal["linear", "threshold", "modulo", "tree_stump", "sine"]
MechanismKind = Literal[
    "linear",
    "mlp",
    "gated_mlp",
    "product",
    "piecewise",
    "sine",
    "spatial",
    "random_fourier",
]
ClassAssignmentKind = Literal["rank", "value", "nested", "dirichlet", "multilabel_score"]


@dataclass(frozen=True)
class PriorConfig:
    min_rows: int = 128
    max_rows: int = 2048
    log_rows: bool = True
    replay_small_prob: float = 0.05
    min_features: int = 4
    max_features: int = 64
    min_hidden_nodes: int = 8
    max_hidden_nodes: int = 128
    max_parents: int = 4
    prior: PriorMode = "bag"
    scm_weight: float = 0.90
    tree_weight: float = 0.05
    gp_weight: float = 0.05
    task: TaskMode = "mixed"
    classification_prob: float = 0.75
    max_classes: int = 160
    many_class_prob: float = 0.25
    class_assignment_kinds: tuple[ClassAssignmentKind, ...] = (
        "rank",
        "value",
        "nested",
        "dirichlet",
        "multilabel_score",
    )
    balanced_class_prob: float = 0.10
    ordered_label_prob: float = 0.20
    permute_label_prob: float = 0.80
    valid_split_attempts: int = 20
    categorical_feature_prob: float = 0.25
    max_categories: int = 64
    high_cardinality_categorical_prob: float = 0.10
    max_high_cardinality_categories: int = 1000
    temporal_prob: float = 0.20
    ood_prob: float = 0.25
    dynamic_scm_prob: float = 0.35
    max_lag: int = 5
    rolling_window_min: int = 2
    rolling_window_max: int = 16
    event_shock_prob: float = 0.20
    concept_shift_prob: float = 0.25
    min_train_fraction: float = 0.50
    max_train_fraction: float = 0.80
    noise_min: float = 0.005
    noise_max: float = 0.20
    node_dims: tuple[int, ...] = (1, 2, 4, 8, 16)
    node_dim_probs: tuple[float, ...] = (0.30, 0.20, 0.25, 0.20, 0.05)
    node_standardize_prob: float = 0.80
    node_clip_prob: float = 0.80
    node_clip_quantile: float = 0.001
    marginal_warp_prob: float = 0.50
    mcar_missing_prob: float = 0.70
    mar_missing_prob: float = 0.30
    mnar_missing_prob: float = 0.15
    max_missing_rate: float = 0.60
    outlier_column_prob: float = 0.20
    max_outlier_rate: float = 0.10
    difficulty_filter: bool = True
    min_classification_probe_margin: float = 0.02
    max_classification_probe_accuracy: float = 0.995
    min_regression_probe_r2: float = 0.01
    max_regression_probe_r2: float = 0.995
    prototype_root_prob: float = 0.25
    correlated_root_prob: float = 0.35
    graph_layouts: tuple[GraphLayout, ...] = (
        "layered",
        "erdos_renyi",
        "barabasi_albert",
        "random_tree",
        "reverse_tree",
        "random_cauchy",
        "block",
        "hub",
        "small_world",
    )
    aggregation_kinds: tuple[AggregationKind, ...] = (
        "sum",
        "mean",
        "max",
        "product",
        "logexp",
        "gated_sum",
        "attention",
        "quadratic",
    )
    activation_kinds: tuple[ActivationKind, ...] = (
        "identity",
        "relu",
        "tanh",
        "sigmoid",
        "gelu",
        "softplus",
        "abs",
        "log_signed",
        "sine",
        "cosine",
    )
    source_kinds: tuple[SourceKind, ...] = (
        "normal",
        "uniform",
        "beta",
        "zipf",
        "mixed",
        "trend",
        "cycle",
        "multi_sine",
        "ar1",
        "seasonal_ar",
        "event",
    )
    edge_weight_kinds: tuple[EdgeWeightKind, ...] = ("gaussian", "uniform", "lognormal", "cauchy")
    edge_mechanism_kinds: tuple[EdgeMechanismKind, ...] = (
        "linear",
        "threshold",
        "modulo",
        "tree_stump",
        "sine",
    )
    mechanism_kinds: tuple[MechanismKind, ...] = (
        "linear",
        "mlp",
        "gated_mlp",
        "product",
        "piecewise",
        "sine",
        "spatial",
        "random_fourier",
    )
    propagate_chunk_size: int = 4096
    device: str = "cpu"
    tree_min_depth: int = 2
    tree_max_depth: int = 6
    tree_min_estimators: int = 3
    tree_max_estimators: int = 12
    gp_num_basis: int = 256
    gp_lengthscale_min: float = 0.20
    gp_lengthscale_max: float = 3.00
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.min_rows < 8 or self.min_rows > self.max_rows:
            raise ValueError("row range must satisfy 8 <= min_rows <= max_rows")
        if self.min_features < 1 or self.min_features > self.max_features:
            raise ValueError("feature range must satisfy 1 <= min_features <= max_features")
        if self.min_hidden_nodes < 1 or self.min_hidden_nodes > self.max_hidden_nodes:
            raise ValueError(
                "hidden node range must satisfy 1 <= min_hidden_nodes <= max_hidden_nodes"
            )
        if self.max_parents < 1:
            raise ValueError("max_parents must be positive")
        if self.scm_weight < 0.0 or self.tree_weight < 0.0 or self.gp_weight < 0.0:
            raise ValueError("prior weights must be non-negative")
        if self.scm_weight + self.tree_weight + self.gp_weight <= 0.0:
            raise ValueError("at least one prior weight must be positive")
        if self.max_classes < 2:
            raise ValueError("max_classes must be at least 2")
        if self.max_categories < 2:
            raise ValueError("max_categories must be at least 2")
        if self.max_high_cardinality_categories < self.max_categories:
            raise ValueError("max_high_cardinality_categories must be >= max_categories")
        if not 0.0 <= self.classification_prob <= 1.0:
            raise ValueError("classification_prob must be in [0, 1]")
        if not 0.0 <= self.many_class_prob <= 1.0:
            raise ValueError("many_class_prob must be in [0, 1]")
        if not 0.0 <= self.balanced_class_prob <= 1.0:
            raise ValueError("balanced_class_prob must be in [0, 1]")
        if not 0.0 <= self.ordered_label_prob <= 1.0:
            raise ValueError("ordered_label_prob must be in [0, 1]")
        if not 0.0 <= self.permute_label_prob <= 1.0:
            raise ValueError("permute_label_prob must be in [0, 1]")
        if self.valid_split_attempts < 1:
            raise ValueError("valid_split_attempts must be positive")
        if not 0.0 <= self.categorical_feature_prob <= 1.0:
            raise ValueError("categorical_feature_prob must be in [0, 1]")
        if not 0.0 <= self.high_cardinality_categorical_prob <= 1.0:
            raise ValueError("high_cardinality_categorical_prob must be in [0, 1]")
        if not 0.0 <= self.temporal_prob <= 1.0:
            raise ValueError("temporal_prob must be in [0, 1]")
        if not 0.0 <= self.ood_prob <= 1.0:
            raise ValueError("ood_prob must be in [0, 1]")
        if not 0.0 <= self.dynamic_scm_prob <= 1.0:
            raise ValueError("dynamic_scm_prob must be in [0, 1]")
        if self.max_lag < 1:
            raise ValueError("max_lag must be positive")
        if self.rolling_window_min < 1 or self.rolling_window_min > self.rolling_window_max:
            raise ValueError("rolling window range must satisfy 1 <= min <= max")
        if not 0.0 <= self.event_shock_prob <= 1.0:
            raise ValueError("event_shock_prob must be in [0, 1]")
        if not 0.0 <= self.concept_shift_prob <= 1.0:
            raise ValueError("concept_shift_prob must be in [0, 1]")
        if not 0.0 <= self.replay_small_prob <= 1.0:
            raise ValueError("replay_small_prob must be in [0, 1]")
        if not 0.0 < self.min_train_fraction < self.max_train_fraction < 1.0:
            raise ValueError("train fractions must satisfy 0 < min < max < 1")
        if not 0.0 < self.noise_min <= self.noise_max:
            raise ValueError("noise range must satisfy 0 < noise_min <= noise_max")
        if len(self.node_dims) != len(self.node_dim_probs):
            raise ValueError("node_dims and node_dim_probs must have the same length")
        if any(dim < 1 for dim in self.node_dims):
            raise ValueError("node_dims must contain positive integers")
        if any(prob < 0.0 for prob in self.node_dim_probs) or sum(self.node_dim_probs) <= 0.0:
            raise ValueError("node_dim_probs must be non-negative with positive total mass")
        if not 0.0 <= self.node_standardize_prob <= 1.0:
            raise ValueError("node_standardize_prob must be in [0, 1]")
        if not 0.0 <= self.node_clip_prob <= 1.0:
            raise ValueError("node_clip_prob must be in [0, 1]")
        if not 0.0 <= self.node_clip_quantile < 0.5:
            raise ValueError("node_clip_quantile must be in [0, 0.5)")
        if not 0.0 <= self.marginal_warp_prob <= 1.0:
            raise ValueError("marginal_warp_prob must be in [0, 1]")
        if not 0.0 <= self.mcar_missing_prob <= 1.0:
            raise ValueError("mcar_missing_prob must be in [0, 1]")
        if not 0.0 <= self.mar_missing_prob <= 1.0:
            raise ValueError("mar_missing_prob must be in [0, 1]")
        if not 0.0 <= self.mnar_missing_prob <= 1.0:
            raise ValueError("mnar_missing_prob must be in [0, 1]")
        if not 0.0 <= self.max_missing_rate <= 1.0:
            raise ValueError("max_missing_rate must be in [0, 1]")
        if not 0.0 <= self.outlier_column_prob <= 1.0:
            raise ValueError("outlier_column_prob must be in [0, 1]")
        if not 0.0 <= self.max_outlier_rate <= 1.0:
            raise ValueError("max_outlier_rate must be in [0, 1]")
        if not 0.0 <= self.min_classification_probe_margin <= 1.0:
            raise ValueError("min_classification_probe_margin must be in [0, 1]")
        if not 0.0 <= self.max_classification_probe_accuracy <= 1.0:
            raise ValueError("max_classification_probe_accuracy must be in [0, 1]")
        if self.min_regression_probe_r2 > self.max_regression_probe_r2:
            raise ValueError("regression probe R2 thresholds must satisfy min <= max")
        if not 0.0 <= self.prototype_root_prob <= 1.0:
            raise ValueError("prototype_root_prob must be in [0, 1]")
        if not 0.0 <= self.correlated_root_prob <= 1.0:
            raise ValueError("correlated_root_prob must be in [0, 1]")
        if self.propagate_chunk_size < 1:
            raise ValueError("propagate_chunk_size must be positive")
        if self.tree_min_depth < 1 or self.tree_min_depth > self.tree_max_depth:
            raise ValueError("tree depth range must satisfy 1 <= min <= max")
        if self.tree_min_estimators < 1 or self.tree_min_estimators > self.tree_max_estimators:
            raise ValueError("tree estimator range must satisfy 1 <= min <= max")
        if self.gp_num_basis < 1:
            raise ValueError("gp_num_basis must be positive")
        if not 0.0 < self.gp_lengthscale_min <= self.gp_lengthscale_max:
            raise ValueError("GP lengthscale range must satisfy 0 < min <= max")
        self._validate_non_empty("graph_layouts", self.graph_layouts)
        self._validate_non_empty("aggregation_kinds", self.aggregation_kinds)
        self._validate_non_empty("activation_kinds", self.activation_kinds)
        self._validate_non_empty("source_kinds", self.source_kinds)
        self._validate_non_empty("edge_weight_kinds", self.edge_weight_kinds)
        self._validate_non_empty("edge_mechanism_kinds", self.edge_mechanism_kinds)
        self._validate_non_empty("mechanism_kinds", self.mechanism_kinds)
        self._validate_non_empty("class_assignment_kinds", self.class_assignment_kinds)

    @staticmethod
    def _validate_non_empty(name: str, values: tuple[object, ...]) -> None:
        if len(values) == 0:
            raise ValueError(f"{name} must be non-empty")
