from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from plurel.bipartite import (
    assign_cluster_at_levels,
    get_probs_at_levels,
    sample_bipartite_assignments,
)


@dataclass(frozen=True)
class EdgePriorSpec:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    null_rate: float = 0.0


class EdgePriorSampler(ABC):
    @abstractmethod
    def sample(
        self,
        size_a: int,
        size_b: int,
        params: dict[str, Any],
        rng: np.random.Generator,
        child_timestamps: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        raise NotImplementedError


class HSBMSampler(EdgePriorSampler):
    def sample(
        self,
        size_a: int,
        size_b: int,
        params: dict[str, Any],
        rng: np.random.Generator,
        child_timestamps: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        hierarchy_a = params.get("hierarchy_a")
        hierarchy_b = params.get("hierarchy_b")
        if hierarchy_a is None or hierarchy_b is None:
            raise ValueError("HSBM params require hierarchy_a and hierarchy_b")
        return (
            sample_bipartite_assignments(
                size_a=size_a,
                size_b=size_b,
                hierarchy_a=list(hierarchy_a),
                hierarchy_b=list(hierarchy_b),
            ),
            None,
        )


class ChungLuSampler(EdgePriorSampler):
    def sample(
        self,
        size_a: int,
        size_b: int,
        params: dict[str, Any],
        rng: np.random.Generator,
        child_timestamps: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        _validate_sizes(size_a=size_a, size_b=size_b)
        gamma = float(params.get("gamma", 2.0))
        weights = 1.0 / np.power(np.arange(1, size_a + 1, dtype=float), gamma)
        weights /= weights.sum()
        return rng.choice(size_a, size=size_b, replace=True, p=weights).astype(np.int64), None


class ErdosRenyiSampler(EdgePriorSampler):
    def sample(
        self,
        size_a: int,
        size_b: int,
        params: dict[str, Any],
        rng: np.random.Generator,
        child_timestamps: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        _validate_sizes(size_a=size_a, size_b=size_b)
        return rng.integers(0, size_a, size=size_b, dtype=np.int64), None


class DCSBMSampler(EdgePriorSampler):
    def sample(
        self,
        size_a: int,
        size_b: int,
        params: dict[str, Any],
        rng: np.random.Generator,
        child_timestamps: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        _validate_sizes(size_a=size_a, size_b=size_b)
        hierarchy_a = params.get("hierarchy_a")
        hierarchy_b = params.get("hierarchy_b")
        if hierarchy_a is None or hierarchy_b is None:
            raise ValueError("DCSBM params require hierarchy_a and hierarchy_b")

        alpha = float(params.get("theta_alpha", 2.0))
        beta = float(params.get("theta_beta", 5.0))
        theta = np.clip(rng.beta(alpha, beta, size=size_a), 1e-12, None)
        strength = float(params.get("degree_correction_strength", 1.0))
        strength = float(np.clip(strength, 0.0, 1.0))
        theta = np.power(theta, strength)

        hierarchy_a = list(hierarchy_a)
        hierarchy_b = list(hierarchy_b)
        cluster_at_levels_a = assign_cluster_at_levels(num_nodes=size_a, hierarchy=hierarchy_a)
        cluster_at_levels_b = assign_cluster_at_levels(num_nodes=size_b, hierarchy=hierarchy_b)
        probs_at_levels = get_probs_at_levels(hierarchy_a=hierarchy_a, hierarchy_b=hierarchy_b)
        log_p_at_levels = [np.log(p) for p in probs_at_levels]
        log_theta = np.log(theta)[:, None]

        chunk_memory_bytes = int(params.get("chunk_memory_bytes", 100_000_000))
        bytes_per_cell = 8
        chunk = max(1, min(size_b, chunk_memory_bytes // max(1, size_a * bytes_per_cell)))

        parent_idx = np.empty(size_b, dtype=np.int64)
        for b_start in range(0, size_b, chunk):
            b_end = min(b_start + chunk, size_b)
            chunk_width = b_end - b_start
            log_p = np.zeros((size_a, chunk_width), dtype=np.float64)
            for level_idx, log_p_level in enumerate(log_p_at_levels):
                log_p += log_p_level[
                    cluster_at_levels_a[:, level_idx][:, None],
                    cluster_at_levels_b[b_start:b_end, level_idx][None, :],
                ]
            log_p += log_theta
            log_p -= log_p.max(axis=0, keepdims=True)
            p = np.exp(log_p)
            p /= p.sum(axis=0, keepdims=True)
            cdf = np.cumsum(p, axis=0)
            u = rng.uniform(0.0, 1.0, size=(1, chunk_width))
            parent_idx[b_start:b_end] = (cdf >= u).argmax(axis=0)
        return parent_idx, None


class TPASampler(EdgePriorSampler):
    def sample(
        self,
        size_a: int,
        size_b: int,
        params: dict[str, Any],
        rng: np.random.Generator,
        child_timestamps: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        _validate_sizes(size_a=size_a, size_b=size_b)
        alpha = float(params.get("alpha", 1.0))
        beta = float(params.get("beta", 0.0))
        epsilon = float(params.get("epsilon", 1.0))

        if child_timestamps is None:
            order = np.arange(size_b)
            timestamp_values = np.arange(size_b, dtype=float)
        else:
            timestamp_values = _timestamps_to_float(child_timestamps)
            if timestamp_values.shape != (size_b,):
                raise ValueError("child_timestamps must have shape (size_b,)")
            order = np.argsort(timestamp_values, kind="mergesort")

        first_seen = np.full(size_a, np.nan, dtype=float)
        degree = np.zeros(size_a, dtype=float)
        parent_idx = np.empty(size_b, dtype=np.int64)

        for child_pos in order:
            current_time = timestamp_values[child_pos]
            degree_term = np.power(degree + epsilon, alpha)
            if beta > 0.0 and np.isfinite(current_time):
                observed = np.isfinite(first_seen)
                recency = np.ones(size_a, dtype=float)
                recency[observed] = np.exp(
                    -beta * np.maximum(current_time - first_seen[observed], 0.0)
                )
                weights = degree_term * recency
            else:
                weights = degree_term
            if weights.sum() <= 0.0 or not np.all(np.isfinite(weights)):
                weights = np.ones(size_a, dtype=float)
            weights = weights / weights.sum()
            parent = int(rng.choice(size_a, p=weights))
            parent_idx[child_pos] = parent
            degree[parent] += 1.0
            if not np.isfinite(first_seen[parent]):
                first_seen[parent] = current_time
        return parent_idx, None


class StructuralNullDecorator(EdgePriorSampler):
    def __init__(self, sampler: EdgePriorSampler):
        self.sampler = sampler

    def sample(
        self,
        size_a: int,
        size_b: int,
        params: dict[str, Any],
        rng: np.random.Generator,
        child_timestamps: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        parent_idx, null_mask = self.sampler.sample(
            size_a=size_a,
            size_b=size_b,
            params=params,
            rng=rng,
            child_timestamps=child_timestamps,
        )
        null_rate = float(params.get("null_rate", 0.0))
        if null_rate <= 0.0:
            return parent_idx, null_mask
        sampled_null_mask = rng.random(size_b) < min(null_rate, 1.0)
        if null_mask is not None:
            sampled_null_mask |= null_mask
        return parent_idx, sampled_null_mask


TOPOLOGY_PRIOR_REGISTRY: dict[str, type[EdgePriorSampler]] = {
    "hsbm": HSBMSampler,
    "erdos_renyi": ErdosRenyiSampler,
    "chung_lu": ChungLuSampler,
    "dcsbm": DCSBMSampler,
    "tpa": TPASampler,
}


def default_hsbm_spec(hierarchy_a: list[int], hierarchy_b: list[int]) -> EdgePriorSpec:
    return EdgePriorSpec(
        kind="hsbm",
        params={"hierarchy_a": list(hierarchy_a), "hierarchy_b": list(hierarchy_b)},
        null_rate=0.0,
    )


def _validate_sizes(size_a: int, size_b: int) -> None:
    if size_a <= 0:
        raise ValueError("size_a must be positive")
    if size_b < 0:
        raise ValueError("size_b must be non-negative")


def _timestamps_to_float(timestamps: np.ndarray) -> np.ndarray:
    array = np.asarray(timestamps)
    if np.issubdtype(array.dtype, np.datetime64):
        return array.astype("datetime64[ns]").astype(float)
    return array.astype(float, copy=False)
