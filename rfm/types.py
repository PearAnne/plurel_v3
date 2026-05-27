from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

TaskType = Literal["classification", "regression"]
FeatureType = Literal["numerical", "categorical"]
PriorType = Literal["scm", "tree", "gp", "bag", "tabicl", "tabpfn_v1", "real", "file"]


@dataclass(frozen=True)
class DatasetMeta:
    prior_type: PriorType
    num_rows: int
    num_features: int
    task_type: TaskType
    train_size: int
    num_classes: int | None
    temporal: bool
    ood: bool
    feature_nodes: tuple[int, ...]
    target_node: int
    dag_edges: tuple[tuple[int, int], ...]
    categorical_features: tuple[int, ...]
    graph_layout: str | None = None
    source_kinds: tuple[str, ...] = ()
    aggregation_kinds: tuple[str, ...] = ()
    mechanism_kinds: tuple[str, ...] = ()
    edge_mechanism_kinds: tuple[str, ...] = ()
    dynamic_nodes: tuple[int, ...] = ()
    node_dim: int = 1
    feature_components: tuple[int, ...] = ()
    target_component: int = 0


@dataclass(frozen=True)
class SyntheticDataset:
    x: NDArray[np.float32]
    y: NDArray[np.integer] | NDArray[np.floating]
    feature_types: tuple[FeatureType, ...]
    meta: DatasetMeta


@dataclass(frozen=True)
class SyntheticBatch:
    x: NDArray[np.float32]
    y: NDArray[np.float32]
    row_mask: NDArray[np.bool_]
    feature_mask: NDArray[np.bool_]
    feature_is_categorical: NDArray[np.bool_]
    train_sizes: NDArray[np.int64]
    task_types: tuple[TaskType, ...]
    num_classes: NDArray[np.int64]
