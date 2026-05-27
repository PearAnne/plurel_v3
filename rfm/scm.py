from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from rfm.config import (
    ActivationKind,
    AggregationKind,
    EdgeMechanismKind,
    GraphLayout,
    MechanismKind,
    PriorConfig,
    SourceKind,
)


@dataclass(frozen=True)
class EdgeMechanism:
    kind: EdgeMechanismKind
    weight: float
    bias: float
    threshold: float
    frequency: float
    phase: float
    modulus: float
    left_value: float
    right_value: float


@dataclass(frozen=True)
class NodeMechanism:
    kind: MechanismKind
    aggregation: AggregationKind
    activation: ActivationKind
    weights: NDArray[np.float64]
    bias: float
    noise_scale: float
    edge_mechanisms: tuple[EdgeMechanism, ...] = ()
    hidden_weights: NDArray[np.float64] | None = None
    hidden_output: NDArray[np.float64] | None = None
    frequency: float = 1.0
    phase: float = 0.0
    center: NDArray[np.float64] | None = None
    width: float = 1.0
    fourier_weights: NDArray[np.float64] | None = None
    fourier_phases: NDArray[np.float64] | None = None
    fourier_output: NDArray[np.float64] | None = None
    gate_weights: NDArray[np.float64] | None = None
    interaction_matrix: NDArray[np.float64] | None = None
    piecewise_knots: NDArray[np.float64] | None = None
    piecewise_slopes: NDArray[np.float64] | None = None
    attention_temperature: float = 1.0
    dynamic: bool = False
    lag: int = 1
    rolling_window: int = 2
    ar_weight: float = 0.0
    parent_lag_weight: float = 0.0
    rolling_weight: float = 0.0
    time_weight: float = 0.0
    shock_scale: float = 0.0
    concept_shift_multiplier: float = 1.0


@dataclass(frozen=True)
class ExogenousContext:
    row_context: NDArray[np.float64]
    parent_context: NDArray[np.float64]
    topology_context: NDArray[np.float64]
    edge_messages: NDArray[np.float64] | None = None

    @property
    def num_rows(self) -> int:
        return self.row_context.shape[0]

    def stacked(self) -> NDArray[np.float64]:
        pieces = [
            self.row_context.astype(np.float64, copy=False),
            self.parent_context.astype(np.float64, copy=False),
            self.topology_context.astype(np.float64, copy=False),
        ]
        if self.edge_messages is not None and self.edge_messages.size > 0:
            pieces.append(self.edge_messages.astype(np.float64, copy=False))
        return np.concatenate(pieces, axis=1)

    @property
    def matrix(self) -> NDArray[np.float64]:
        return self.stacked()


@dataclass(frozen=True)
class SCMSpec:
    parents: tuple[tuple[int, ...], ...]
    edge_weights: tuple[tuple[float, ...], ...]
    mechanisms: tuple[NodeMechanism | None, ...]
    source_kinds: tuple[SourceKind | None, ...]
    graph_layout: GraphLayout
    node_dim: int
    exogenous_nodes: tuple[int, ...] = ()
    num_exogenous: int = 0
    exogenous_is_root: tuple[bool, ...] = ()

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (parent, child) for child, parents in enumerate(self.parents) for parent in parents
        )

    @property
    def root_source_kinds(self) -> tuple[str, ...]:
        return tuple(source for source in self.source_kinds if source is not None)

    @property
    def aggregation_kinds(self) -> tuple[str, ...]:
        return tuple(
            mechanism.aggregation for mechanism in self.mechanisms if mechanism is not None
        )

    @property
    def mechanism_kinds(self) -> tuple[str, ...]:
        return tuple(mechanism.kind for mechanism in self.mechanisms if mechanism is not None)

    @property
    def edge_mechanism_kinds(self) -> tuple[str, ...]:
        return tuple(
            edge_mechanism.kind
            for mechanism in self.mechanisms
            if mechanism is not None
            for edge_mechanism in mechanism.edge_mechanisms
        )

    @property
    def dynamic_nodes(self) -> tuple[int, ...]:
        return tuple(
            node
            for node, mechanism in enumerate(self.mechanisms)
            if mechanism is not None and mechanism.dynamic
        )


class SCMGenerator:
    def __init__(
        self,
        rng: np.random.Generator,
        config: PriorConfig,
    ) -> None:
        self.rng = rng
        self.config = config
        self.device = torch.device(config.device)

    def sample_spec(
        self,
        num_nodes: int,
        node_dim: int,
        prefer_temporal: bool = False,
        enable_dynamic: bool = True,
    ) -> SCMSpec:
        if num_nodes < 3:
            raise ValueError("num_nodes must be at least 3")
        if node_dim < 1:
            raise ValueError("node_dim must be positive")

        graph_layout = self._sample_choice(self.config.graph_layouts)
        parents = self._sample_parents(num_nodes, graph_layout)
        parents = self._ensure_target_has_parents(parents)

        mechanisms: list[NodeMechanism | None] = []
        source_kinds: list[SourceKind | None] = []
        edge_weights: list[tuple[float, ...]] = []
        temporal_source_used = False

        for node, node_parents in enumerate(parents):
            if not node_parents:
                source_kind = self._sample_source_kind(prefer_temporal and not temporal_source_used)
                temporal_source_used = temporal_source_used or source_kind in (
                    "trend",
                    "cycle",
                    "multi_sine",
                    "ar1",
                    "seasonal_ar",
                    "event",
                )
                mechanisms.append(None)
                source_kinds.append(source_kind)
                edge_weights.append(())
                continue

            mechanism = self._sample_mechanism(
                len(node_parents), node_dim=node_dim, enable_dynamic=enable_dynamic
            )
            mechanisms.append(mechanism)
            source_kinds.append(None)
            edge_weights.append(
                tuple(edge_mechanism.weight for edge_mechanism in mechanism.edge_mechanisms)
            )

        return SCMSpec(
            parents=tuple(parents),
            edge_weights=tuple(edge_weights),
            mechanisms=tuple(mechanisms),
            source_kinds=tuple(source_kinds),
            graph_layout=graph_layout,
            node_dim=node_dim,
        )

    def sample_spec_with_exogenous(
        self,
        num_nodes: int,
        num_exogenous: int,
        node_dim: int,
        prefer_temporal: bool = False,
        enable_dynamic: bool = True,
    ) -> SCMSpec:
        if num_nodes < 1:
            raise ValueError("num_nodes must be at least 1")
        if num_exogenous < 1:
            raise ValueError("num_exogenous must be at least 1")
        if node_dim < 1:
            raise ValueError("node_dim must be positive")

        graph_layout = self._sample_choice(self.config.graph_layouts)
        internal_parents = self._sample_parents(num_nodes, graph_layout)
        internal_parents = self._ensure_target_has_parents(internal_parents)

        parents: list[tuple[int, ...]] = [() for _ in range(num_exogenous)]
        mechanisms: list[NodeMechanism | None] = [None for _ in range(num_exogenous)]
        source_kinds: list[SourceKind | None] = [None for _ in range(num_exogenous)]
        edge_weights: list[tuple[float, ...]] = [() for _ in range(num_exogenous)]
        exogenous_is_root = tuple(True for _ in range(num_exogenous))

        temporal_source_used = False
        for node_parents in internal_parents:
            shifted: list[int] = []
            for parent in node_parents:
                if self.rng.random() < 0.65:
                    shifted.append(int(self.rng.integers(0, num_exogenous)))
                shifted.append(parent + num_exogenous)
            shifted = sorted(set(shifted))
            if not shifted:
                shifted = [int(self.rng.integers(0, num_exogenous))]
            parents.append(tuple(shifted))

            mechanism = self._sample_mechanism(
                len(shifted), node_dim=node_dim, enable_dynamic=enable_dynamic
            )
            mechanisms.append(mechanism)
            source_kinds.append(None)
            edge_weights.append(
                tuple(edge_mechanism.weight for edge_mechanism in mechanism.edge_mechanisms)
            )
            temporal_source_used = temporal_source_used or False

        return SCMSpec(
            parents=tuple(parents),
            edge_weights=tuple(edge_weights),
            mechanisms=tuple(mechanisms),
            source_kinds=tuple(source_kinds),
            graph_layout=graph_layout,
            node_dim=node_dim,
            num_exogenous=num_exogenous,
            exogenous_is_root=exogenous_is_root,
        )

    def sample_values(
        self,
        spec: SCMSpec,
        num_rows: int,
        train_size: int,
        temporal: bool,
        ood: bool,
        exogenous: ExogenousContext | None = None,
    ) -> NDArray[np.float64]:
        values = torch.zeros(
            (num_rows, len(spec.parents), spec.node_dim), dtype=torch.float64, device=self.device
        )
        source_nodes = [
            node
            for node, node_parents in enumerate(spec.parents)
            if len(node_parents) == 0 and node >= spec.num_exogenous
        ]
        source_values = self._sample_source_values(
            spec=spec,
            source_nodes=source_nodes,
            num_rows=num_rows,
            train_size=train_size,
            temporal=temporal,
            ood=ood,
        )

        if exogenous is not None and spec.num_exogenous > 0:
            self._inject_exogenous(values, spec, exogenous)

        for node in range(len(spec.parents)):
            node_parents = spec.parents[node]
            if node < spec.num_exogenous:
                continue
            if not node_parents:
                values[:, node, :] = source_values[node]
                continue

            mechanism = spec.mechanisms[node]
            if mechanism is None:
                raise ValueError(f"missing mechanism for non-root node {node}")
            parent_values = values[:, list(node_parents), :]
            values[:, node, :] = self._compute_node(
                parent_values=parent_values,
                edge_weights=spec.edge_weights[node],
                mechanism=mechanism,
                train_size=train_size,
                ood=ood,
            )

        return values.detach().cpu().numpy()

    def _inject_exogenous(
        self,
        values: torch.Tensor,
        spec: SCMSpec,
        exogenous: ExogenousContext,
    ) -> None:
        if exogenous.num_rows != values.shape[0]:
            raise ValueError("exogenous row count does not match num_rows")
        stacked = exogenous.stacked().astype(np.float64, copy=False)
        cursor = 0
        for node in range(spec.num_exogenous):
            stop = min(cursor + spec.node_dim, stacked.shape[1])
            block = stacked[:, cursor:stop]
            if block.shape[1] < spec.node_dim:
                pad = np.zeros((block.shape[0], spec.node_dim - block.shape[1]), dtype=np.float64)
                block = np.concatenate([block, pad], axis=1)
            values[:, node, :] = torch.as_tensor(block, dtype=torch.float64, device=self.device)
            cursor += spec.node_dim

    def _sample_parents(self, num_nodes: int, graph_layout: GraphLayout) -> list[tuple[int, ...]]:
        if graph_layout == "layered":
            return self._sample_layered_parents(num_nodes)
        if graph_layout == "erdos_renyi":
            return self._sample_erdos_renyi_parents(num_nodes)
        if graph_layout == "barabasi_albert":
            return self._sample_barabasi_albert_parents(num_nodes)
        if graph_layout == "random_tree":
            return self._sample_tree_parents(num_nodes, reverse=False)
        if graph_layout == "reverse_tree":
            return self._sample_tree_parents(num_nodes, reverse=True)
        if graph_layout == "random_cauchy":
            return self._sample_random_cauchy_parents(num_nodes)
        if graph_layout == "block":
            return self._sample_block_parents(num_nodes)
        if graph_layout == "hub":
            return self._sample_hub_parents(num_nodes)
        if graph_layout == "small_world":
            return self._sample_small_world_parents(num_nodes)
        raise ValueError(f"unknown graph layout {graph_layout}")

    def _sample_layered_parents(self, num_nodes: int) -> list[tuple[int, ...]]:
        depth = int(self.rng.integers(2, min(7, num_nodes + 1)))
        sizes = [1] * depth
        for _ in range(num_nodes - depth):
            sizes[int(self.rng.integers(0, depth))] += 1

        layers: list[list[int]] = []
        cursor = 0
        for size in sizes:
            layers.append(list(range(cursor, cursor + size)))
            cursor += size

        parents: list[tuple[int, ...]] = [() for _ in range(num_nodes)]
        for layer_idx, layer in enumerate(layers[1:], start=1):
            previous = [node for layer_nodes in layers[:layer_idx] for node in layer_nodes]
            direct_previous = layers[layer_idx - 1]
            for node in layer:
                candidates = direct_previous if self.rng.random() < 0.8 else previous
                parents[node] = self._sample_parent_subset(candidates)
        return parents

    def _sample_erdos_renyi_parents(self, num_nodes: int) -> list[tuple[int, ...]]:
        parents: list[tuple[int, ...]] = [()]
        for node in range(1, num_nodes):
            upper = max(0.05, min(0.65, self.config.max_parents / max(node, 1)))
            p_edge = float(self.rng.uniform(0.05, upper))
            candidates = [candidate for candidate in range(node) if self.rng.random() < p_edge]
            parents.append(self._limit_parent_count(candidates))
        return parents

    def _sample_barabasi_albert_parents(self, num_nodes: int) -> list[tuple[int, ...]]:
        parents: list[tuple[int, ...]] = [()]
        degrees = np.ones(num_nodes, dtype=np.float64)
        for node in range(1, num_nodes):
            max_count = min(self.config.max_parents, node, 4)
            parent_count = int(self.rng.integers(1, max_count + 1))
            probabilities = degrees[:node] / degrees[:node].sum()
            node_parents = sorted(
                self.rng.choice(node, size=parent_count, replace=False, p=probabilities).tolist()
            )
            parents.append(tuple(int(parent) for parent in node_parents))
            degrees[node] += parent_count
            degrees[node_parents] += 1.0
        return parents

    def _sample_tree_parents(self, num_nodes: int, reverse: bool) -> list[tuple[int, ...]]:
        parents: list[tuple[int, ...]] = [()]
        for node in range(1, num_nodes):
            if reverse:
                low = max(0, node - max(2, self.config.max_parents * 2))
                parent = int(self.rng.integers(low, node))
            else:
                parent = int(self.rng.integers(0, node))
            node_parents = [parent]
            for candidate in range(node):
                if candidate != parent and self.rng.random() < 0.04:
                    node_parents.append(candidate)
            parents.append(self._limit_parent_count(node_parents))
        return parents

    def _sample_random_cauchy_parents(self, num_nodes: int) -> list[tuple[int, ...]]:
        parents: list[tuple[int, ...]] = [()]
        global_bias = float(np.clip(self.rng.standard_cauchy(), -4.0, 4.0))
        source_bias = np.clip(self.rng.standard_cauchy(num_nodes), -4.0, 4.0)
        target_bias = np.clip(self.rng.standard_cauchy(num_nodes), -4.0, 4.0)
        for node in range(1, num_nodes):
            candidates: list[int] = []
            for candidate in range(node):
                logit = global_bias + source_bias[candidate] + target_bias[node]
                probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
                if self.rng.random() < probability:
                    candidates.append(candidate)
            parents.append(self._limit_parent_count(candidates))
        return parents

    def _sample_block_parents(self, num_nodes: int) -> list[tuple[int, ...]]:
        num_blocks = int(self.rng.integers(2, min(7, num_nodes + 1)))
        block_ids = np.floor(np.linspace(0, num_blocks, num_nodes, endpoint=False)).astype(np.int64)
        parents: list[tuple[int, ...]] = [()]
        for node in range(1, num_nodes):
            same_block = [
                candidate for candidate in range(node) if block_ids[candidate] == block_ids[node]
            ]
            previous_block = [
                candidate for candidate in range(node) if block_ids[candidate] < block_ids[node]
            ]
            candidates: list[int] = []
            for candidate in same_block:
                if self.rng.random() < 0.35:
                    candidates.append(candidate)
            for candidate in previous_block:
                if self.rng.random() < 0.08:
                    candidates.append(candidate)
            if not candidates:
                pool = same_block or list(range(node))
                candidates.append(int(self.rng.choice(pool)))
            parents.append(self._limit_parent_count(candidates))
        return parents

    def _sample_hub_parents(self, num_nodes: int) -> list[tuple[int, ...]]:
        hub_count = int(self.rng.integers(1, min(5, num_nodes)))
        hubs = list(range(hub_count))
        parents: list[tuple[int, ...]] = [() for _ in range(hub_count)]
        for node in range(hub_count, num_nodes):
            candidates = [hub for hub in hubs if hub < node and self.rng.random() < 0.75]
            for candidate in range(hub_count, node):
                if self.rng.random() < 0.06:
                    candidates.append(candidate)
            if not candidates:
                candidates.append(int(self.rng.integers(0, node)))
            parents.append(self._limit_parent_count(candidates))
        return parents

    def _sample_small_world_parents(self, num_nodes: int) -> list[tuple[int, ...]]:
        parents: list[tuple[int, ...]] = [()]
        local_radius = int(self.rng.integers(2, min(8, num_nodes + 1)))
        for node in range(1, num_nodes):
            candidates = [
                candidate
                for candidate in range(max(0, node - local_radius), node)
                if self.rng.random() < 0.55
            ]
            for candidate in range(0, max(0, node - local_radius)):
                if self.rng.random() < 0.03:
                    candidates.append(candidate)
            if not candidates:
                candidates.append(int(self.rng.integers(max(0, node - local_radius), node)))
            parents.append(self._limit_parent_count(candidates))
        return parents

    def _ensure_target_has_parents(self, parents: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
        target_node = len(parents) - 1
        if parents[target_node]:
            return parents
        parent_count = int(self.rng.integers(1, min(self.config.max_parents, target_node) + 1))
        parents[target_node] = tuple(
            sorted(self.rng.choice(target_node, size=parent_count, replace=False).tolist())
        )
        return parents

    def _sample_parent_subset(self, candidates: list[int]) -> tuple[int, ...]:
        if not candidates:
            return ()
        max_count = min(self.config.max_parents, len(candidates))
        parent_count = int(self.rng.integers(1, max_count + 1))
        return tuple(sorted(self.rng.choice(candidates, size=parent_count, replace=False).tolist()))

    def _limit_parent_count(self, candidates: list[int]) -> tuple[int, ...]:
        unique_candidates = sorted(set(int(candidate) for candidate in candidates))
        if len(unique_candidates) <= self.config.max_parents:
            return tuple(unique_candidates)
        return tuple(
            sorted(
                self.rng.choice(
                    unique_candidates, size=self.config.max_parents, replace=False
                ).tolist()
            )
        )

    def _sample_source_kind(self, prefer_temporal: bool) -> SourceKind:
        temporal_kinds = tuple(
            kind
            for kind in self.config.source_kinds
            if kind in ("trend", "cycle", "multi_sine", "ar1", "seasonal_ar", "event")
        )
        if prefer_temporal and temporal_kinds:
            return self._sample_choice(temporal_kinds)
        return self._sample_choice(self.config.source_kinds)

    def _source_kind_for_node(self, spec: SCMSpec, node: int, temporal: bool) -> SourceKind:
        source_kind = spec.source_kinds[node]
        if source_kind is None:
            raise ValueError(f"node {node} is not a source node")
        if (
            temporal
            and node == 0
            and source_kind not in ("trend", "cycle", "multi_sine", "ar1", "seasonal_ar", "event")
        ):
            temporal_kinds = tuple(
                kind
                for kind in self.config.source_kinds
                if kind in ("trend", "cycle", "multi_sine", "ar1", "seasonal_ar", "event")
            )
            if temporal_kinds:
                return self._sample_choice(temporal_kinds)
        return source_kind

    def _sample_source_values(
        self,
        spec: SCMSpec,
        source_nodes: list[int],
        num_rows: int,
        train_size: int,
        temporal: bool,
        ood: bool,
        exogenous: ExogenousContext | None = None,
    ) -> dict[int, torch.Tensor]:
        if not source_nodes:
            return {}

        exogenous_matrix: NDArray[np.float64] | None = None
        if exogenous is not None:
            exogenous_matrix = _standardize_exogenous(exogenous.matrix)

        raw_by_node: dict[int, NDArray[np.float64]] = {}
        flat_columns: list[NDArray[np.float64]] = []
        for node in source_nodes:
            if exogenous_matrix is not None and node in spec.exogenous_nodes:
                column = min(node, exogenous_matrix.shape[1] - 1)
                node_values = exogenous_matrix[:, column : column + 1].astype(np.float64)
                if spec.node_dim > 1:
                    node_values = np.repeat(node_values, spec.node_dim, axis=1)
                raw_by_node[node] = node_values
                flat_columns.append(node_values)
                continue
            source_kind = self._source_kind_for_node(spec, node, temporal)
            node_values = np.column_stack(
                [
                    self._sample_raw_source_column(num_rows=num_rows, source_kind=source_kind)
                    for _ in range(spec.node_dim)
                ]
            ).astype(np.float64)
            raw_by_node[node] = node_values
            flat_columns.append(node_values)

        flat = np.concatenate(flat_columns, axis=1)
        if flat.shape[1] > 1 and self.rng.random() < self.config.prototype_root_prob:
            flat = self._apply_prototype_mixing(flat)
        if flat.shape[1] > 1 and self.rng.random() < self.config.correlated_root_prob:
            flat = self._apply_correlated_root_noise(flat)

        if ood:
            shift = self.rng.normal(0.5, 0.2, size=flat.shape[1])
            scale = self.rng.uniform(0.7, 1.8, size=flat.shape[1])
            flat[train_size:] = flat[train_size:] * scale + shift
        flat = flat + self.rng.normal(0.0, 1e-4, size=flat.shape)
        for col in range(flat.shape[1]):
            flat[:, col] = _standardize_np(flat[:, col], train_size=train_size if ood else None)

        source_values: dict[int, torch.Tensor] = {}
        cursor = 0
        for node in source_nodes:
            node_values = flat[:, cursor : cursor + spec.node_dim]
            source_values[node] = torch.as_tensor(
                node_values, dtype=torch.float64, device=self.device
            )
            cursor += spec.node_dim
        return source_values

    def _apply_prototype_mixing(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        component_count = int(self.rng.integers(2, min(9, values.shape[0] + 1)))
        prototypes = self.rng.normal(0.0, 1.0, size=(component_count, values.shape[1]))
        assignments = self.rng.integers(0, component_count, size=values.shape[0])
        noise_scale = float(np.exp(self.rng.uniform(np.log(0.01), np.log(0.7))))
        return (
            values + prototypes[assignments] + self.rng.normal(0.0, noise_scale, size=values.shape)
        )

    def _apply_correlated_root_noise(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        factor_count = int(self.rng.integers(1, min(5, values.shape[1] + 1)))
        factors = self.rng.normal(0.0, 1.0, size=(values.shape[0], factor_count))
        loadings = self.rng.normal(0.0, 0.8, size=(factor_count, values.shape[1]))
        return values + factors @ loadings

    def _sample_raw_source_column(
        self,
        num_rows: int,
        source_kind: SourceKind,
    ) -> NDArray[np.float64]:
        if source_kind == "mixed":
            source_kind = self._sample_choice(("normal", "uniform", "beta", "zipf"))

        if source_kind == "normal":
            raw = self.rng.normal(0.0, self.rng.uniform(0.5, 2.0), size=num_rows)
        elif source_kind == "uniform":
            low, high = sorted(self.rng.uniform(-3.0, 3.0, size=2))
            raw = self.rng.uniform(low, high, size=num_rows)
        elif source_kind == "beta":
            alpha = float(self.rng.uniform(0.5, 5.0))
            beta = float(self.rng.uniform(0.5, 5.0))
            raw = self.rng.beta(alpha, beta, size=num_rows)
        elif source_kind == "zipf":
            raw = np.minimum(self.rng.zipf(float(self.rng.uniform(1.5, 4.0)), size=num_rows), 20.0)
        elif source_kind == "trend":
            slope = float(self.rng.uniform(-2.0, 2.0))
            raw = slope * np.linspace(-1.0, 1.0, num_rows) + self.rng.normal(
                0.0, 0.1, size=num_rows
            )
        elif source_kind == "cycle":
            time = np.linspace(0.0, 1.0, num_rows)
            frequency = float(self.rng.uniform(1.0, 25.0))
            phase = float(self.rng.uniform(0.0, 2.0 * np.pi))
            raw = np.sin(2.0 * np.pi * frequency * time + phase) + self.rng.normal(
                0.0, 0.05, size=num_rows
            )
        elif source_kind == "multi_sine":
            time = np.linspace(0.0, 1.0, num_rows)
            raw = np.zeros(num_rows, dtype=np.float64)
            for _ in range(int(self.rng.integers(2, 7))):
                amplitude = float(self.rng.uniform(0.2, 1.5))
                frequency = float(np.exp(self.rng.uniform(np.log(0.25), np.log(150.0))))
                phase = float(self.rng.uniform(0.0, 2.0 * np.pi))
                raw = raw + amplitude * np.sin(2.0 * np.pi * frequency * time + phase)
            raw = raw + self.rng.normal(0.0, 0.05, size=num_rows)
        elif source_kind == "ar1":
            rho = float(self.rng.uniform(0.0, 0.95))
            raw = np.zeros(num_rows, dtype=np.float64)
            noise = self.rng.normal(0.0, 1.0, size=num_rows)
            for idx in range(1, num_rows):
                raw[idx] = rho * raw[idx - 1] + noise[idx]
        elif source_kind == "seasonal_ar":
            rho = float(self.rng.uniform(0.2, 0.95))
            period = float(self.rng.uniform(4.0, max(5.0, num_rows / 3.0)))
            raw = np.zeros(num_rows, dtype=np.float64)
            noise = self.rng.normal(0.0, 0.5, size=num_rows)
            for idx in range(1, num_rows):
                seasonal = np.sin(2.0 * np.pi * idx / period)
                raw[idx] = rho * raw[idx - 1] + seasonal + noise[idx]
        elif source_kind == "event":
            raw = self.rng.normal(0.0, 0.3, size=num_rows)
            idx = np.arange(num_rows, dtype=np.float64)
            for _ in range(int(self.rng.integers(1, 5))):
                center = float(self.rng.integers(0, num_rows))
                width = float(self.rng.uniform(1.0, max(2.0, num_rows / 10.0)))
                amplitude = float(self.rng.normal(0.0, 2.0))
                raw = raw + amplitude * np.exp(-((idx - center) ** 2) / (2.0 * width * width))
        else:
            raise ValueError(f"unknown source kind {source_kind}")

        return raw.astype(np.float64, copy=True)

    def _sample_mechanism(
        self, parent_count: int, node_dim: int, enable_dynamic: bool = True
    ) -> NodeMechanism:
        choices = list(self.config.mechanism_kinds)
        if parent_count < 2:
            choices = [kind for kind in choices if kind not in ("spatial", "piecewise")]
        kind = self._sample_choice(tuple(choices))
        aggregation = self._sample_choice(self.config.aggregation_kinds)
        activation = self._sample_choice(self.config.activation_kinds)
        weights = self.rng.normal(0.0, 1.0, size=parent_count)
        bias = float(self.rng.normal(0.0, 0.5))
        noise_scale = float(
            np.exp(self.rng.uniform(np.log(self.config.noise_min), np.log(self.config.noise_max)))
        )
        input_dim = parent_count * node_dim
        edge_mechanisms = tuple(self._sample_edge_mechanism() for _ in range(parent_count))
        dynamic = bool(enable_dynamic and self.rng.random() < self.config.dynamic_scm_prob)
        dynamic_kwargs = self._sample_dynamic_kwargs(dynamic)

        if kind in ("mlp", "gated_mlp"):
            hidden_dim = int(self.rng.integers(4, 17))
            return NodeMechanism(
                kind=kind,
                aggregation=aggregation,
                activation=activation,
                weights=weights,
                bias=bias,
                noise_scale=noise_scale,
                edge_mechanisms=edge_mechanisms,
                hidden_weights=self.rng.normal(0.0, 1.0, size=(input_dim, hidden_dim)),
                hidden_output=self.rng.normal(0.0, 1.0, size=(hidden_dim, node_dim)),
                gate_weights=self.rng.normal(0.0, 1.0, size=input_dim)
                if kind == "gated_mlp"
                else None,
                **dynamic_kwargs,
            )
        if kind == "sine":
            return NodeMechanism(
                kind=kind,
                aggregation=aggregation,
                activation=activation,
                weights=weights,
                bias=bias,
                noise_scale=noise_scale,
                edge_mechanisms=edge_mechanisms,
                frequency=float(np.exp(self.rng.uniform(np.log(0.05), np.log(500.0)))),
                phase=float(self.rng.uniform(0.0, 2.0 * np.pi)),
                **dynamic_kwargs,
            )
        if kind == "spatial":
            return NodeMechanism(
                kind=kind,
                aggregation=aggregation,
                activation=activation,
                weights=weights,
                bias=bias,
                noise_scale=noise_scale,
                edge_mechanisms=edge_mechanisms,
                center=self.rng.normal(0.0, 1.0, size=2),
                width=float(np.exp(self.rng.uniform(np.log(0.2), np.log(3.0)))),
                **dynamic_kwargs,
            )
        if kind == "random_fourier":
            basis = int(self.rng.integers(16, 65))
            return NodeMechanism(
                kind=kind,
                aggregation=aggregation,
                activation=activation,
                weights=weights,
                bias=bias,
                noise_scale=noise_scale,
                edge_mechanisms=edge_mechanisms,
                fourier_weights=self.rng.normal(
                    0.0, self.rng.uniform(0.05, 80.0), size=(input_dim, basis)
                ),
                fourier_phases=self.rng.uniform(0.0, 2.0 * np.pi, size=basis),
                fourier_output=self.rng.normal(0.0, 1.0 / np.sqrt(basis), size=(basis, node_dim)),
                **dynamic_kwargs,
            )
        if kind == "piecewise":
            pieces = int(self.rng.integers(3, 8))
            return NodeMechanism(
                kind=kind,
                aggregation=aggregation,
                activation=activation,
                weights=weights,
                bias=bias,
                noise_scale=noise_scale,
                edge_mechanisms=edge_mechanisms,
                piecewise_knots=np.sort(self.rng.normal(0.0, 1.0, size=pieces)),
                piecewise_slopes=self.rng.normal(0.0, 1.0, size=pieces + 1),
                **dynamic_kwargs,
            )
        return NodeMechanism(
            kind=kind,
            aggregation=aggregation,
            activation=activation,
            weights=weights,
            bias=bias,
            noise_scale=noise_scale,
            edge_mechanisms=edge_mechanisms,
            interaction_matrix=self.rng.normal(0.0, 0.5, size=(input_dim, input_dim)),
            gate_weights=self.rng.normal(0.0, 1.0, size=input_dim),
            attention_temperature=float(np.exp(self.rng.uniform(np.log(0.2), np.log(3.0)))),
            **dynamic_kwargs,
        )

    def _sample_edge_mechanism(self) -> EdgeMechanism:
        kind = self._sample_choice(self.config.edge_mechanism_kinds)
        return EdgeMechanism(
            kind=kind,
            weight=self._sample_edge_weight(),
            bias=float(self.rng.normal(0.0, 0.5)),
            threshold=float(self.rng.normal(0.0, 1.0)),
            frequency=float(np.exp(self.rng.uniform(np.log(0.1), np.log(80.0)))),
            phase=float(self.rng.uniform(0.0, 2.0 * np.pi)),
            modulus=float(np.exp(self.rng.uniform(np.log(0.2), np.log(5.0)))),
            left_value=float(self.rng.normal(-0.5, 1.0)),
            right_value=float(self.rng.normal(0.5, 1.0)),
        )

    def _sample_dynamic_kwargs(self, dynamic: bool) -> dict[str, float | int | bool]:
        if not dynamic:
            return {"dynamic": False}
        return {
            "dynamic": True,
            "lag": int(self.rng.integers(1, self.config.max_lag + 1)),
            "rolling_window": int(
                self.rng.integers(
                    self.config.rolling_window_min, self.config.rolling_window_max + 1
                )
            ),
            "ar_weight": float(self.rng.uniform(-0.75, 0.75)),
            "parent_lag_weight": float(self.rng.uniform(-0.8, 0.8)),
            "rolling_weight": float(self.rng.uniform(-0.6, 0.6)),
            "time_weight": float(self.rng.uniform(-1.0, 1.0)),
            "shock_scale": float(self.rng.uniform(0.0, 1.5))
            if self.rng.random() < self.config.event_shock_prob
            else 0.0,
            "concept_shift_multiplier": (
                float(self.rng.uniform(0.5, 1.8))
                if self.rng.random() < self.config.concept_shift_prob
                else 1.0
            ),
        }

    def _compute_node(
        self,
        parent_values: torch.Tensor,
        edge_weights: tuple[float, ...],
        mechanism: NodeMechanism,
        train_size: int,
        ood: bool,
    ) -> torch.Tensor:
        raw = self._compute_static_node(parent_values, edge_weights, mechanism)
        raw = self._apply_activation(raw, mechanism.activation)
        if mechanism.dynamic:
            raw = self._apply_dynamic_scm(
                raw=raw,
                parent_values=parent_values,
                mechanism=mechanism,
                train_size=train_size,
                ood=ood,
            )
        noise = torch.as_tensor(
            self.rng.normal(0.0, mechanism.noise_scale, size=tuple(raw.shape)),
            dtype=torch.float64,
            device=self.device,
        )
        return self._normalize_node(raw + noise)

    def _compute_static_node(
        self,
        parent_values: torch.Tensor,
        edge_weights: tuple[float, ...],
        mechanism: NodeMechanism,
    ) -> torch.Tensor:
        chunk_size = self.config.propagate_chunk_size
        if mechanism.dynamic or parent_values.shape[0] <= chunk_size:
            return self._compute_node_raw(parent_values, edge_weights, mechanism)

        chunks = []
        for start in range(0, parent_values.shape[0], chunk_size):
            stop = min(start + chunk_size, parent_values.shape[0])
            chunks.append(
                self._compute_node_raw(parent_values[start:stop], edge_weights, mechanism)
            )
        return torch.cat(chunks, dim=0)

    def _compute_node_raw(
        self,
        parent_values: torch.Tensor,
        edge_weights: tuple[float, ...],
        mechanism: NodeMechanism,
    ) -> torch.Tensor:
        edge_values = self._apply_edge_mechanisms(parent_values, edge_weights, mechanism)
        base = self._aggregate(edge_values, mechanism)
        flat_parent_values = parent_values.reshape(parent_values.shape[0], -1)
        flat_edge_values = edge_values.reshape(edge_values.shape[0], -1)

        if mechanism.kind == "linear":
            raw = base + mechanism.bias
        elif mechanism.kind == "mlp":
            if mechanism.hidden_weights is None or mechanism.hidden_output is None:
                raise ValueError("MLP mechanism is missing weights")
            hidden_weights = torch.as_tensor(
                mechanism.hidden_weights, dtype=torch.float64, device=self.device
            )
            hidden_output = torch.as_tensor(
                mechanism.hidden_output, dtype=torch.float64, device=self.device
            )
            hidden = torch.tanh(flat_edge_values @ hidden_weights + mechanism.bias)
            raw = hidden @ hidden_output
        elif mechanism.kind == "gated_mlp":
            if (
                mechanism.hidden_weights is None
                or mechanism.hidden_output is None
                or mechanism.gate_weights is None
            ):
                raise ValueError("gated MLP mechanism is missing weights")
            hidden_weights = torch.as_tensor(
                mechanism.hidden_weights, dtype=torch.float64, device=self.device
            )
            hidden_output = torch.as_tensor(
                mechanism.hidden_output, dtype=torch.float64, device=self.device
            )
            gate_weights = torch.as_tensor(
                mechanism.gate_weights, dtype=torch.float64, device=self.device
            )
            gate = torch.sigmoid(flat_parent_values @ gate_weights).unsqueeze(1)
            hidden = torch.tanh(flat_edge_values @ hidden_weights + mechanism.bias)
            raw = gate * (hidden @ hidden_output) + (1.0 - gate) * base
        elif mechanism.kind == "product":
            raw = torch.prod(torch.tanh(edge_values) + 1.2, dim=1) + mechanism.bias
        elif mechanism.kind == "piecewise":
            if mechanism.piecewise_knots is None or mechanism.piecewise_slopes is None:
                raise ValueError("piecewise mechanism is missing knots")
            knots = torch.as_tensor(
                mechanism.piecewise_knots, dtype=torch.float64, device=self.device
            )
            slopes = torch.as_tensor(
                mechanism.piecewise_slopes, dtype=torch.float64, device=self.device
            )
            bins = torch.bucketize(base, knots)
            raw = slopes[bins] * base + mechanism.bias
        elif mechanism.kind == "sine":
            raw = torch.sin(mechanism.frequency * base + mechanism.phase)
        elif mechanism.kind == "spatial":
            if mechanism.center is None:
                raise ValueError("spatial mechanism is missing center")
            center = torch.as_tensor(mechanism.center, dtype=torch.float64, device=self.device)
            diff = flat_parent_values[:, :2] - center
            spatial = torch.exp(
                -torch.sum(diff * diff, dim=1) / (2.0 * mechanism.width * mechanism.width)
            )
            raw = spatial.unsqueeze(1).expand(-1, base.shape[1])
        elif mechanism.kind == "random_fourier":
            if (
                mechanism.fourier_weights is None
                or mechanism.fourier_phases is None
                or mechanism.fourier_output is None
            ):
                raise ValueError("random Fourier mechanism is missing weights")
            fourier_weights = torch.as_tensor(
                mechanism.fourier_weights, dtype=torch.float64, device=self.device
            )
            fourier_phases = torch.as_tensor(
                mechanism.fourier_phases, dtype=torch.float64, device=self.device
            )
            fourier_output = torch.as_tensor(
                mechanism.fourier_output, dtype=torch.float64, device=self.device
            )
            features = torch.cos(flat_edge_values @ fourier_weights + fourier_phases)
            raw = features @ fourier_output + mechanism.bias
        else:
            raise ValueError(f"unknown mechanism {mechanism.kind}")

        return raw

    def _apply_edge_mechanisms(
        self,
        parent_values: torch.Tensor,
        edge_weights: tuple[float, ...],
        mechanism: NodeMechanism,
    ) -> torch.Tensor:
        if len(mechanism.edge_mechanisms) != parent_values.shape[1]:
            raise ValueError("edge mechanism count does not match parent count")
        transformed = []
        for parent_idx, edge_mechanism in enumerate(mechanism.edge_mechanisms):
            values = parent_values[:, parent_idx, :]
            if edge_mechanism.kind == "linear":
                out = values + edge_mechanism.bias
            elif edge_mechanism.kind == "threshold":
                out = (values > edge_mechanism.threshold).to(values.dtype)
            elif edge_mechanism.kind == "modulo":
                out = torch.remainder(
                    values * edge_mechanism.frequency + edge_mechanism.phase, edge_mechanism.modulus
                )
                out = out - 0.5 * edge_mechanism.modulus
            elif edge_mechanism.kind == "tree_stump":
                left = torch.full_like(values, edge_mechanism.left_value)
                right = torch.full_like(values, edge_mechanism.right_value)
                out = torch.where(values <= edge_mechanism.threshold, left, right)
            elif edge_mechanism.kind == "sine":
                out = torch.sin(edge_mechanism.frequency * values + edge_mechanism.phase)
            else:
                raise ValueError(f"unknown edge mechanism {edge_mechanism.kind}")
            out = out * edge_weights[parent_idx]
            transformed.append(out.unsqueeze(1))
        return torch.cat(transformed, dim=1)

    def _aggregate(self, values: torch.Tensor, mechanism: NodeMechanism) -> torch.Tensor:
        aggregation = mechanism.aggregation
        clipped = torch.clamp(values, -1e4, 1e4)
        if aggregation == "sum":
            return clipped.sum(dim=1)
        if aggregation == "mean":
            return clipped.mean(dim=1)
        if aggregation == "max":
            return clipped.max(dim=1).values
        if aggregation == "product":
            return torch.prod(torch.tanh(clipped) + 1.2, dim=1)
        if aggregation == "logexp":
            return torch.logsumexp(clipped, dim=1)
        if aggregation == "gated_sum":
            if mechanism.gate_weights is None:
                gate_weights = torch.ones(
                    clipped.shape[1] * clipped.shape[2], dtype=torch.float64, device=self.device
                )
            else:
                gate_weights = torch.as_tensor(
                    mechanism.gate_weights, dtype=torch.float64, device=self.device
                )
            gates = torch.sigmoid(clipped.reshape(clipped.shape[0], -1) * gate_weights)
            gates = gates.reshape_as(clipped)
            return (gates * clipped).sum(dim=1)
        if aggregation == "attention":
            scores = clipped / mechanism.attention_temperature
            attn = torch.softmax(scores, dim=1)
            return (attn * clipped).sum(dim=1)
        if aggregation == "quadratic":
            if mechanism.interaction_matrix is None:
                interaction = torch.eye(
                    clipped.shape[1] * clipped.shape[2], dtype=torch.float64, device=self.device
                )
            else:
                interaction = torch.as_tensor(
                    mechanism.interaction_matrix, dtype=torch.float64, device=self.device
                )
            flat = clipped.reshape(clipped.shape[0], -1)
            quadratic = torch.einsum("bi,ij,bj->b", flat, interaction, flat) / max(1, flat.shape[1])
            return quadratic.unsqueeze(1).expand(-1, clipped.shape[2])
        raise ValueError(f"unknown aggregation {aggregation}")

    def _apply_dynamic_scm(
        self,
        raw: torch.Tensor,
        parent_values: torch.Tensor,
        mechanism: NodeMechanism,
        train_size: int,
        ood: bool,
    ) -> torch.Tensor:
        out = torch.empty_like(raw)
        lag = min(mechanism.lag, max(1, raw.shape[0] - 1))
        rolling_window = min(mechanism.rolling_window, max(1, raw.shape[0]))
        time = torch.linspace(-1.0, 1.0, raw.shape[0], dtype=raw.dtype, device=raw.device)
        seasonal = torch.sin(2.0 * torch.pi * (lag + 1) * time + mechanism.phase)
        shocks = torch.zeros_like(raw)
        if mechanism.shock_scale > 0.0:
            centers = self.rng.choice(
                raw.shape[0], size=int(self.rng.integers(1, 4)), replace=False
            )
            width = float(self.rng.uniform(2.0, max(3.0, raw.shape[0] / 12.0)))
            idx = torch.arange(raw.shape[0], dtype=raw.dtype, device=raw.device).unsqueeze(1)
            for center in centers:
                shocks = shocks + mechanism.shock_scale * torch.exp(
                    -((idx - float(center)) ** 2) / (2.0 * width * width)
                )

        parent_lagged = torch.zeros_like(raw)
        parent_mean = parent_values.mean(dim=1)
        if raw.shape[0] > lag:
            parent_lagged[lag:] = parent_mean[:-lag]

        for row in range(raw.shape[0]):
            lagged_self = out[row - lag] if row >= lag else raw[row]
            start = max(0, row - rolling_window)
            rolling_self = out[start:row].mean(dim=0) if row > start else raw[row]
            shift_multiplier = (
                mechanism.concept_shift_multiplier if ood and row >= train_size else 1.0
            )
            out[row] = (
                raw[row] * shift_multiplier
                + mechanism.ar_weight * lagged_self
                + mechanism.parent_lag_weight * parent_lagged[row]
                + mechanism.rolling_weight * rolling_self
                + mechanism.time_weight * seasonal[row]
                + shocks[row]
            )
        return out

    def _apply_activation(self, values: torch.Tensor, activation: ActivationKind) -> torch.Tensor:
        if activation == "identity":
            return values
        if activation == "relu":
            return torch.relu(values)
        if activation == "tanh":
            return torch.tanh(values)
        if activation == "sigmoid":
            return torch.sigmoid(values)
        if activation == "gelu":
            return torch.nn.functional.gelu(values)
        if activation == "softplus":
            return torch.nn.functional.softplus(values)
        if activation == "abs":
            return torch.abs(values)
        if activation == "log_signed":
            return torch.log1p(torch.abs(values)) * torch.sign(values)
        if activation == "sine":
            return torch.sin(values)
        if activation == "cosine":
            return torch.cos(values)
        raise ValueError(f"unknown activation {activation}")

    def _normalize_node(self, values: torch.Tensor) -> torch.Tensor:
        if self.rng.random() < self.config.node_clip_prob:
            q = self.config.node_clip_quantile
            if q > 0.0:
                lower = torch.quantile(values, q, dim=0)
                upper = torch.quantile(values, 1.0 - q, dim=0)
                values = torch.minimum(torch.maximum(values, lower), upper)
        if self.rng.random() < self.config.node_standardize_prob:
            values = _standardize_tensor(values)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("non-finite SCM node values")
        return values

    def _sample_edge_weight(self) -> float:
        kind = self._sample_choice(self.config.edge_weight_kinds)
        if kind == "gaussian":
            return float(self.rng.normal())
        if kind == "uniform":
            return float(self.rng.uniform(-2.0, 2.0))
        if kind == "lognormal":
            sign = -1.0 if self.rng.random() < 0.5 else 1.0
            return float(sign * self.rng.lognormal(mean=0.0, sigma=0.75))
        if kind == "cauchy":
            return float(np.clip(self.rng.standard_cauchy(), -5.0, 5.0))
        raise ValueError(f"unknown edge weight kind {kind}")

    def _sample_choice(self, values: tuple) -> object:
        return values[int(self.rng.integers(0, len(values)))]


def _standardize_exogenous(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    standardized = np.empty_like(matrix, dtype=np.float64)
    for col in range(matrix.shape[1]):
        column = matrix[:, col].astype(np.float64, copy=True)
        std = float(np.std(column))
        if std <= 1e-12 or not np.isfinite(std):
            standardized[:, col] = column - float(np.mean(column))
        else:
            standardized[:, col] = (column - float(np.mean(column))) / std
    return standardized


def _standardize_np(
    values: NDArray[np.float64], train_size: int | None = None
) -> NDArray[np.float64]:
    fit_values = values if train_size is None else values[:train_size]
    std = float(np.std(fit_values))
    if std <= 1e-12 or not np.isfinite(std):
        raise ValueError("constant column generated inside SCM")
    return (values - float(np.mean(fit_values))) / std


def _standardize_tensor(values: torch.Tensor) -> torch.Tensor:
    std = torch.std(values, dim=0, unbiased=False)
    if not bool(torch.isfinite(std).all()) or bool(torch.any(std <= 1e-12)):
        raise ValueError("constant column generated inside SCM")
    return (values - torch.mean(values, dim=0)) / std
