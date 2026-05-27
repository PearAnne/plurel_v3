from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from rfm.rdb.config import RDBPriorConfig
from rfm.rdb.types import (
    AttachmentMode,
    EdgeIntentSpec,
    ExistenceMode,
    ForeignKeySpec,
    MechanismProfile,
    RelationProfileKind,
    SchemaGraph,
)

_PROFILE_TO_ATTACHMENT: dict[RelationProfileKind, AttachmentMode] = {
    "uniform": "uniform",
    "popularity": "hub_preferential",
    "locality": "segment_local",
    "temporal": "temporal_causal",
    "capacity": "uniform",
    "multi_parent": "segment_local",
    "hybrid": "uniform",
}


class MechanismProfileSampler:
    def __init__(self, rng: np.random.Generator, config: RDBPriorConfig) -> None:
        self.rng = rng
        self.config = config
        self.hyper = config.mechanism_hyperprior

    def sample_all(self, schema: SchemaGraph) -> dict[str, MechanismProfile]:
        profiles: dict[str, MechanismProfile] = {}
        for fk in schema.edges:
            intent_spec = schema.edge_intents[fk.key]
            profiles[fk.key] = self.sample_one(fk, intent_spec)
        return profiles

    def sample_one(self, fk: ForeignKeySpec, intent_spec: EdgeIntentSpec) -> MechanismProfile:
        latent_dim = self.config.latent_dim
        attachment = self._sample_attachment(fk, intent_spec)
        existence = self._sample_existence(fk)
        coordination: str = (
            intent_spec.coordination if intent_spec.coordination != "bridge_pair" else "joint_tuple"
        )
        if fk.multi_parent_group is not None:
            coordination = "joint_tuple"

        field_weights = self.rng.normal(
            0.0,
            self.hyper.field_weight_scale / np.sqrt(latent_dim),
            size=latent_dim,
        ).astype(np.float64)
        existence_latent = self.rng.normal(
            0.0,
            self.hyper.existence_latent_scale / np.sqrt(latent_dim),
            size=latent_dim,
        ).astype(np.float64)

        hub_strength = float(self.rng.uniform(*self.hyper.hub_strength_range))
        locality_strength = float(self.rng.uniform(*self.hyper.locality_strength_range))
        temporal_strength = float(self.rng.uniform(*self.hyper.temporal_strength_range))
        compat_strength = float(self.rng.uniform(*self.hyper.compat_strength_range))

        if attachment == "hub_preferential":
            hub_strength *= 1.5
        if attachment == "segment_local":
            locality_strength *= 1.5
        if attachment == "temporal_causal":
            temporal_strength *= 1.5
        if attachment == "bridge_pairing":
            locality_strength *= 1.2
        if fk.multi_parent_group is not None and self.hyper.forced_attachment is None:
            attachment = "segment_local"
            locality_strength *= 2.5

        capacity_mode = fk.mechanism.capacity_mode
        capacity_k = fk.capacity

        return MechanismProfile(
            existence=existence,
            attachment=attachment,
            coordination=coordination,  # type: ignore[arg-type]
            field_weights=tuple(float(v) for v in field_weights),
            temperature=float(
                self.rng.uniform(self.hyper.temperature_min, self.hyper.temperature_max)
            ),
            capacity_mode=capacity_mode,
            existence_bias=self._existence_bias(existence),
            existence_latent_weight=tuple(float(v) for v in existence_latent),
            existence_time_weight=float(self.rng.normal(0.0, self.hyper.existence_time_scale)),
            hub_strength=hub_strength,
            locality_strength=locality_strength,
            temporal_strength=temporal_strength,
            bridge_same_segment_bias=float(self.rng.normal(0.0, 0.5)),
            compat_strength=compat_strength,
            noise_scale=float(self.rng.uniform(0.02, 0.15)),
            capacity_k=capacity_k,
        )

    def apply_profiles(
        self, schema: SchemaGraph, profiles: Mapping[str, MechanismProfile]
    ) -> SchemaGraph:
        edges = tuple(
            replace(fk, mechanism=profiles[fk.key], existence=profiles[fk.key].existence)
            for fk in schema.edges
        )
        return replace(schema, edges=edges)

    def _sample_attachment(self, fk: ForeignKeySpec, intent_spec: EdgeIntentSpec) -> AttachmentMode:
        if self.hyper.forced_attachment is not None:
            return self.hyper.forced_attachment
        if intent_spec.coordination == "bridge_pair":
            return "bridge_pairing"
        if fk.cardinality == "capacity_limited":
            return "uniform"
        if len(self.config.relation_profiles) == 1:
            return _PROFILE_TO_ATTACHMENT[self.config.relation_profiles[0]]
        kind = self.config.relation_profiles[
            int(self.rng.integers(0, len(self.config.relation_profiles)))
        ]
        return _PROFILE_TO_ATTACHMENT[kind]

    def _sample_existence(self, fk: ForeignKeySpec) -> ExistenceMode:
        if fk.existence in ("optional", "sparse"):
            return fk.existence
        if fk.cardinality == "optional" or fk.nullable:
            return "optional"
        if self.hyper.forced_existence is not None:
            return self.hyper.forced_existence
        return "mandatory"

    def _existence_bias(self, existence: ExistenceMode) -> float:
        if existence == "mandatory":
            return self.hyper.mandatory_existence_bias
        if existence == "sparse":
            return self.hyper.sparse_existence_bias
        return self.hyper.optional_existence_bias


class ExistenceGate:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def sample_null_mask(
        self,
        child_latents: NDArray[np.float64],
        child_times: NDArray[np.float64],
        profile: MechanismProfile,
        intent_bias: float = 0.0,
    ) -> NDArray[np.bool_]:
        num_rows = child_latents.shape[0]
        if profile.existence == "mandatory":
            return np.zeros(num_rows, dtype=np.bool_)

        latent_weight = profile.existence_latent_weight_array
        if latent_weight is None:
            latent_term = np.zeros(num_rows, dtype=np.float64)
        else:
            latent_term = child_latents @ latent_weight

        logits = (
            profile.existence_bias
            + latent_term
            + profile.existence_time_weight * child_times
            + intent_bias
            + self.rng.normal(0.0, 0.05, size=num_rows)
        )
        probabilities = _sigmoid(logits / max(profile.temperature, 1e-6))
        if profile.existence == "optional":
            probabilities = np.clip(probabilities, 0.88, 0.97)
        if profile.existence == "sparse":
            probabilities = np.clip(probabilities * 0.35, 0.40, 0.80)
        exists = self.rng.random(num_rows) < probabilities
        return ~exists


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))
