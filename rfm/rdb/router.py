from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rfm.rdb.config import RDBPriorConfig
from rfm.rdb.mechanism import ExistenceGate
from rfm.rdb.multiparent import JointParentTupleSampler
from rfm.rdb.types import ForeignKeySpec, MandatoryFKStats, MechanismProfile, SchemaGraph, TableSpec


class MandatoryFKSatisfactionError(RuntimeError):
    """Raised when a mandatory FK cannot be satisfied under the active policy."""


@dataclass(frozen=True)
class RelationRoutingResult:
    values: dict[str, NDArray[np.float32]]
    null_masks: dict[str, NDArray[np.bool_]]
    fanout: dict[str, NDArray[np.int64]]
    profile_counts: dict[str, int]
    mechanism_counts: dict[str, int]
    mandatory_fk_stats: MandatoryFKStats
    timestamps: dict[str, NDArray[np.float64]]


@dataclass
class RouterState:
    fanout: dict[str, NDArray[np.int64]]
    exposure: dict[str, NDArray[np.float64]]
    recency: dict[str, NDArray[np.float64]]
    dynamic_popularity: dict[str, NDArray[np.float64]]


class CandidateConstraintGate:
    def candidates(
        self,
        fk: ForeignKeySpec,
        child_row: int,
        parent_count: int,
        child_time: float,
        parent_timestamps: NDArray[np.float64],
        fanout: NDArray[np.int64],
        peer_parent_rows: Mapping[str, int] | None = None,
        snapshot_latest_only: bool = False,
    ) -> NDArray[np.int64]:
        candidates = np.arange(parent_count, dtype=np.int64)
        if fk.temporal or snapshot_latest_only:
            candidates = candidates[parent_timestamps[candidates] <= child_time + 1e-12]
        if fk.capacity is not None:
            candidates = candidates[fanout[candidates] < fk.capacity]
        elif fk.cardinality == "one_to_one" or fk.mechanism.capacity_mode == "one_to_one":
            candidates = candidates[fanout[candidates] < 1]
        elif fk.mechanism.capacity_mode == "k_limited" and fk.mechanism.capacity_k is not None:
            candidates = candidates[fanout[candidates] < fk.mechanism.capacity_k]
        if peer_parent_rows is not None and fk.semantic == "bridge_pairs_entities":
            same_parent = peer_parent_rows.get(fk.parent_table)
            if same_parent is not None:
                candidates = candidates[candidates != same_parent]
        if snapshot_latest_only and len(candidates) > 0:
            parent_times = parent_timestamps[candidates]
            latest = float(np.max(parent_times))
            candidates = candidates[np.abs(parent_times - latest) <= 1e-12]
        return candidates


class RelationFieldScorer:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def scores(
        self,
        fk: ForeignKeySpec,
        candidates: NDArray[np.int64],
        child_latent: NDArray[np.float64],
        profile: MechanismProfile,
        parent_latents: NDArray[np.float64],
        child_time: float,
        parent_timestamps: NDArray[np.float64],
        fanout: NDArray[np.int64],
        dynamic_popularity: NDArray[np.float64],
        peer_selections: Mapping[str, int] | None = None,
        row_latents: Mapping[str, NDArray[np.float64]] | None = None,
    ) -> NDArray[np.float64]:
        weights = profile.field_weights_array
        scale = np.sqrt(max(parent_latents.shape[1], 1))
        scores = np.zeros(len(candidates), dtype=np.float64)

        if profile.attachment == "uniform":
            pass
        elif profile.attachment == "hub_preferential":
            scores += profile.hub_strength * np.log1p(
                dynamic_popularity[candidates] + fanout[candidates]
            )
        elif profile.attachment == "temporal_causal":
            lag = np.maximum(0.0, child_time - parent_timestamps[candidates])
            scores += profile.temporal_strength * np.exp(-3.0 * lag)
        elif profile.attachment == "segment_local":
            child_segment = child_latent[0]
            parent_segment = parent_latents[candidates, 0]
            scores -= profile.locality_strength * np.abs(child_segment - parent_segment)
        elif profile.attachment == "bridge_pairing":
            scores += profile.compat_strength * (parent_latents[candidates] @ child_latent) / scale
        else:
            scores += profile.compat_strength * (parent_latents[candidates] @ child_latent) / scale
            scores += profile.compat_strength * (parent_latents[candidates] @ weights) / scale

        if profile.temporal_strength > 0.0 and profile.attachment != "temporal_causal":
            lag = np.maximum(0.0, child_time - parent_timestamps[candidates])
            scores += profile.temporal_strength * np.exp(-3.0 * lag)

        if peer_selections and row_latents is not None:
            for table, peer_row in peer_selections.items():
                if table == fk.parent_table:
                    continue
                peer_latent = row_latents[table][peer_row]
                scores += (
                    profile.compat_strength * (parent_latents[candidates] @ peer_latent) / scale
                )

        if fk.capacity is not None:
            remaining = np.maximum(0, fk.capacity - fanout[candidates])
            scores += 0.8 * remaining / max(float(fk.capacity), 1.0)
        if profile.noise_scale > 0.0:
            scores += self.rng.normal(0.0, profile.noise_scale, size=len(candidates))
        return scores


class TopologyStateUpdater:
    def register(
        self,
        fk: ForeignKeySpec,
        parent_row: int,
        child_time: float,
        state: RouterState,
    ) -> None:
        key = fk.key
        state.fanout[key][parent_row] += 1
        state.exposure[fk.parent_table][parent_row] += 1.0
        state.recency[fk.parent_table][parent_row] = max(
            state.recency[fk.parent_table][parent_row], child_time
        )
        parent_count = state.dynamic_popularity[fk.parent_table].shape[0]
        fanout_term = state.fanout[key] / max(float(state.fanout[key].max(initial=1)), 1.0)
        exposure_term = state.exposure[fk.parent_table] / max(
            float(state.exposure[fk.parent_table].max(initial=1)), 1.0
        )
        recency_term = state.recency[fk.parent_table]
        state.dynamic_popularity[fk.parent_table] = (
            0.45 * fanout_term + 0.35 * exposure_term + 0.20 * recency_term
        )


class TRAPRouter:
    """Topology-aware relation attachment router."""

    def __init__(self, rng: np.random.Generator, config: RDBPriorConfig) -> None:
        self.rng = rng
        self.config = config
        self.existence_gate = ExistenceGate(rng)
        self.constraint_gate = CandidateConstraintGate()
        self.scorer = RelationFieldScorer(rng)
        self.updater = TopologyStateUpdater()
        self.joint_sampler = JointParentTupleSampler(rng, config, self.scorer.scores)

    def route(
        self,
        schema: SchemaGraph,
        timestamps: Mapping[str, NDArray[np.float64]],
        row_latents: Mapping[str, NDArray[np.float64]],
    ) -> RelationRoutingResult:
        return self.route_tables(schema.tables, schema.edges, timestamps, row_latents)

    def route_tables(
        self,
        table_specs: Mapping[str, TableSpec],
        foreign_keys: Sequence[ForeignKeySpec],
        timestamps: Mapping[str, NDArray[np.float64]],
        row_latents: Mapping[str, NDArray[np.float64]],
    ) -> RelationRoutingResult:
        _validate_foreign_key_invariants(table_specs, foreign_keys)
        state = self._init_state(table_specs, foreign_keys)
        values: dict[str, NDArray[np.float32]] = {}
        null_masks: dict[str, NDArray[np.bool_]] = {}
        mechanism_counts: dict[str, int] = {}
        fk_stats = _init_mandatory_fk_stats()

        mutable_timestamps = {
            name: values_array.copy() for name, values_array in timestamps.items()
        }

        for fk in foreign_keys:
            child_latents = row_latents[fk.child_table]
            child_times = mutable_timestamps[fk.child_table]
            null_masks[fk.key] = self.existence_gate.sample_null_mask(
                child_latents=child_latents,
                child_times=child_times,
                profile=fk.mechanism,
            )
            values[fk.key] = np.full(
                table_specs[fk.child_table].row_count, np.nan, dtype=np.float32
            )
            attachment = fk.mechanism.attachment
            mechanism_counts[attachment] = mechanism_counts.get(attachment, 0) + 1

        for child_table in _child_processing_order(table_specs, foreign_keys):
            child_fks = [fk for fk in foreign_keys if fk.child_table == child_table]
            self._route_child_table(
                child_table=child_table,
                child_fks=child_fks,
                table_specs=table_specs,
                timestamps=mutable_timestamps,
                row_latents=row_latents,
                values=values,
                null_masks=null_masks,
                state=state,
                fk_stats=fk_stats,
            )

        profile_counts = dict(mechanism_counts)
        return RelationRoutingResult(
            values=values,
            null_masks=null_masks,
            fanout=state.fanout,
            profile_counts=profile_counts,
            mechanism_counts=mechanism_counts,
            mandatory_fk_stats=_finalize_mandatory_fk_stats(fk_stats),
            timestamps=mutable_timestamps,
        )

    def _route_child_table(
        self,
        child_table: str,
        child_fks: Sequence[ForeignKeySpec],
        table_specs: Mapping[str, TableSpec],
        timestamps: Mapping[str, NDArray[np.float64]],
        row_latents: Mapping[str, NDArray[np.float64]],
        values: dict[str, NDArray[np.float32]],
        null_masks: dict[str, NDArray[np.bool_]],
        state: RouterState,
        fk_stats: dict[str, int],
    ) -> None:
        if len(child_fks) == 0:
            return

        child_count = table_specs[child_table].row_count
        order = np.argsort(timestamps[child_table], kind="mergesort")
        groups = _group_multi_parent(child_fks)
        snapshot_mode = table_specs[child_table].role == "snapshot/state"

        for child_row in order.astype(np.int64):
            child_time = float(timestamps[child_table][child_row])
            child_latent = row_latents[child_table][child_row]
            peer_rows: dict[str, int] = {}

            for group_key, group_fks in groups.items():
                active_fks = [fk for fk in group_fks if not null_masks[fk.key][child_row]]
                if len(active_fks) == 0:
                    continue

                use_joint = len(active_fks) > 1 and (
                    group_key is not None
                    or any(fk.mechanism.coordination == "joint_tuple" for fk in active_fks)
                )

                if use_joint:
                    for fk in active_fks:
                        if fk.existence == "mandatory":
                            fk_stats["mandatory_fk_total"] += 1
                    candidate_lists: list[NDArray[np.int64]] = []
                    for fk in active_fks:
                        candidate_lists.append(
                            self._candidate_list(
                                fk=fk,
                                child_row=int(child_row),
                                parent_count=table_specs[fk.parent_table].row_count,
                                child_time=child_time,
                                parent_timestamps=timestamps[fk.parent_table],
                                fanout=state.fanout[fk.key],
                                peer_parent_rows=peer_rows,
                                snapshot_latest_only=snapshot_mode,
                            )
                        )
                    if any(len(candidates) == 0 for candidates in candidate_lists):
                        fk_stats["multi_parent_candidate_empty_count"] += 1
                        self._handle_unsatisfied_group(
                            active_fks=active_fks,
                            child_table=child_table,
                            child_row=int(child_row),
                            child_time=child_time,
                            table_specs=table_specs,
                            timestamps=timestamps,
                            row_latents=row_latents,
                            values=values,
                            null_masks=null_masks,
                            state=state,
                            peer_rows=peer_rows,
                            snapshot_mode=snapshot_mode,
                            fk_stats=fk_stats,
                            joint=True,
                        )
                        continue
                    choice = self.joint_sampler.sample_tuple(
                        child_row=int(child_row),
                        fks=active_fks,
                        candidate_lists=candidate_lists,
                        child_latent=child_latent,
                        row_latents=row_latents,
                        child_time=child_time,
                        parent_timestamps=timestamps,
                        fanout=state.fanout,
                        dynamic_popularity=state.dynamic_popularity,
                    )
                    for fk, parent_row in zip(active_fks, choice):
                        values[fk.key][child_row] = float(parent_row)
                        peer_rows[fk.parent_table] = parent_row
                        self.updater.register(fk, parent_row, child_time, state)
                    continue

                for fk in active_fks:
                    if fk.existence == "mandatory":
                        fk_stats["mandatory_fk_total"] += 1
                    candidates = self._candidate_list(
                        fk=fk,
                        child_row=int(child_row),
                        parent_count=table_specs[fk.parent_table].row_count,
                        child_time=child_time,
                        parent_timestamps=timestamps[fk.parent_table],
                        fanout=state.fanout[fk.key],
                        peer_parent_rows=peer_rows,
                        snapshot_latest_only=snapshot_mode,
                    )
                    if len(candidates) == 0:
                        self._handle_unsatisfied_group(
                            active_fks=(fk,),
                            child_table=child_table,
                            child_row=int(child_row),
                            child_time=child_time,
                            table_specs=table_specs,
                            timestamps=timestamps,
                            row_latents=row_latents,
                            values=values,
                            null_masks=null_masks,
                            state=state,
                            peer_rows=peer_rows,
                            snapshot_mode=snapshot_mode,
                            fk_stats=fk_stats,
                            joint=False,
                        )
                        continue
                    parent_row = self.joint_sampler.sample_tuple(
                        child_row=int(child_row),
                        fks=(fk,),
                        candidate_lists=(candidates,),
                        child_latent=child_latent,
                        row_latents=row_latents,
                        child_time=child_time,
                        parent_timestamps=timestamps,
                        fanout=state.fanout,
                        dynamic_popularity=state.dynamic_popularity,
                    )[0]
                    values[fk.key][child_row] = float(parent_row)
                    peer_rows[fk.parent_table] = parent_row
                    self.updater.register(fk, parent_row, child_time, state)

    def _candidate_list(
        self,
        fk: ForeignKeySpec,
        child_row: int,
        parent_count: int,
        child_time: float,
        parent_timestamps: NDArray[np.float64],
        fanout: NDArray[np.int64],
        peer_parent_rows: Mapping[str, int] | None,
        snapshot_latest_only: bool,
        relax: bool = False,
    ) -> NDArray[np.int64]:
        if relax:
            candidates = np.arange(parent_count, dtype=np.int64)
            if fk.temporal or snapshot_latest_only:
                candidates = candidates[parent_timestamps[candidates] <= child_time + 1e-12]
            if snapshot_latest_only and len(candidates) > 0:
                parent_times = parent_timestamps[candidates]
                latest = float(np.max(parent_times))
                candidates = candidates[np.abs(parent_times - latest) <= 1e-12]
            return candidates
        return self.constraint_gate.candidates(
            fk=fk,
            child_row=child_row,
            parent_count=parent_count,
            child_time=child_time,
            parent_timestamps=parent_timestamps,
            fanout=fanout,
            peer_parent_rows=peer_parent_rows,
            snapshot_latest_only=snapshot_latest_only,
        )

    def _handle_unsatisfied_group(
        self,
        active_fks: Sequence[ForeignKeySpec],
        child_table: str,
        child_row: int,
        child_time: float,
        table_specs: Mapping[str, TableSpec],
        timestamps: Mapping[str, NDArray[np.float64]],
        row_latents: Mapping[str, NDArray[np.float64]],
        values: dict[str, NDArray[np.float32]],
        null_masks: dict[str, NDArray[np.bool_]],
        state: RouterState,
        peer_rows: dict[str, int],
        snapshot_mode: bool,
        fk_stats: dict[str, int],
        joint: bool,
    ) -> None:
        mandatory_fks = [fk for fk in active_fks if fk.existence == "mandatory"]
        optional_fks = [fk for fk in active_fks if fk.existence != "mandatory"]

        if len(mandatory_fks) == 0:
            for fk in optional_fks:
                null_masks[fk.key][child_row] = True
            return

        if any(fk.temporal for fk in mandatory_fks) and not snapshot_mode:
            if self._try_resample_child_time_assignment(
                mandatory_fks=mandatory_fks,
                child_table=child_table,
                child_row=child_row,
                child_time=child_time,
                table_specs=table_specs,
                timestamps=timestamps,
                row_latents=row_latents,
                values=values,
                state=state,
                peer_rows=peer_rows,
                snapshot_mode=snapshot_mode,
                joint=joint,
                fk_stats=fk_stats,
            ):
                return

        if snapshot_mode and self._try_backoff_assignment(
            mandatory_fks=mandatory_fks,
            child_row=child_row,
            child_time=child_time,
            table_specs=table_specs,
            timestamps=timestamps,
            row_latents=row_latents,
            values=values,
            state=state,
            peer_rows=peer_rows,
            snapshot_mode=snapshot_mode,
            joint=joint,
            fk_stats=fk_stats,
        ):
            if joint:
                fk_stats["joint_sampler_backoff_count"] += 1
            return

        policy = self.config.mandatory_fk_policy
        if policy == "resample_child_time":
            parent_times = [
                float(np.max(timestamps[fk.parent_table]))
                if len(timestamps[fk.parent_table]) > 0
                else child_time
                for fk in mandatory_fks
            ]
            new_time = max(parent_times + [child_time])
            child_time = new_time
            candidate_lists = [
                self._candidate_list(
                    fk=fk,
                    child_row=child_row,
                    parent_count=table_specs[fk.parent_table].row_count,
                    child_time=child_time,
                    parent_timestamps=timestamps[fk.parent_table],
                    fanout=state.fanout[fk.key],
                    peer_parent_rows=peer_rows,
                    snapshot_latest_only=snapshot_mode,
                )
                for fk in mandatory_fks
            ]
            if joint and any(len(candidates) == 0 for candidates in candidate_lists):
                self._fail_or_null_mandatory(
                    mandatory_fks=mandatory_fks,
                    optional_fks=optional_fks,
                    child_row=child_row,
                    null_masks=null_masks,
                    fk_stats=fk_stats,
                )
                return
            if not joint and len(candidate_lists[0]) == 0:
                self._fail_or_null_mandatory(
                    mandatory_fks=mandatory_fks,
                    optional_fks=optional_fks,
                    child_row=child_row,
                    null_masks=null_masks,
                    fk_stats=fk_stats,
                )
                return
            child_latent = row_latents[child_table][child_row]
            choice = self.joint_sampler.sample_tuple(
                child_row=child_row,
                fks=tuple(mandatory_fks),
                candidate_lists=tuple(candidate_lists),
                child_latent=child_latent,
                row_latents=row_latents,
                child_time=child_time,
                parent_timestamps=timestamps,
                fanout=state.fanout,
                dynamic_popularity=state.dynamic_popularity,
            )
            for fk, parent_row in zip(mandatory_fks, choice):
                timestamps[child_table][child_row] = new_time
                fk_stats["mandatory_fk_timestamp_resample_count"] += 1
                values[fk.key][child_row] = float(parent_row)
                peer_rows[fk.parent_table] = parent_row
                self.updater.register(fk, parent_row, child_time, state)
            return

        if policy == "backoff_parent_pool":
            fk_stats["mandatory_fk_backoff_count"] += len(mandatory_fks)
            candidate_lists = [
                self._candidate_list(
                    fk=fk,
                    child_row=child_row,
                    parent_count=table_specs[fk.parent_table].row_count,
                    child_time=child_time,
                    parent_timestamps=timestamps[fk.parent_table],
                    fanout=state.fanout[fk.key],
                    peer_parent_rows=peer_rows,
                    snapshot_latest_only=snapshot_mode,
                    relax=True,
                )
                for fk in mandatory_fks
            ]
            if joint and any(len(candidates) == 0 for candidates in candidate_lists):
                self._fail_or_null_mandatory(
                    mandatory_fks=mandatory_fks,
                    optional_fks=optional_fks,
                    child_row=child_row,
                    null_masks=null_masks,
                    fk_stats=fk_stats,
                )
                return
            if not joint and len(candidate_lists[0]) == 0:
                self._fail_or_null_mandatory(
                    mandatory_fks=mandatory_fks,
                    optional_fks=optional_fks,
                    child_row=child_row,
                    null_masks=null_masks,
                    fk_stats=fk_stats,
                )
                return
            child_latent = row_latents[child_table][child_row]
            choice = self.joint_sampler.sample_tuple(
                child_row=child_row,
                fks=tuple(mandatory_fks),
                candidate_lists=tuple(candidate_lists),
                child_latent=child_latent,
                row_latents=row_latents,
                child_time=child_time,
                parent_timestamps=timestamps,
                fanout=state.fanout,
                dynamic_popularity=state.dynamic_popularity,
            )
            for fk, parent_row in zip(mandatory_fks, choice):
                values[fk.key][child_row] = float(parent_row)
                peer_rows[fk.parent_table] = parent_row
                self.updater.register(fk, parent_row, child_time, state)
            return

        self._fail_or_null_mandatory(
            mandatory_fks=mandatory_fks,
            optional_fks=optional_fks,
            child_row=child_row,
            null_masks=null_masks,
            fk_stats=fk_stats,
        )

    def _fail_or_null_mandatory(
        self,
        mandatory_fks: Sequence[ForeignKeySpec],
        optional_fks: Sequence[ForeignKeySpec],
        child_row: int,
        null_masks: dict[str, NDArray[np.bool_]],
        fk_stats: dict[str, int],
    ) -> None:
        fk_stats["mandatory_fk_unsatisfied"] += len(mandatory_fks)
        for fk in optional_fks:
            null_masks[fk.key][child_row] = True
        raise MandatoryFKSatisfactionError(
            f"mandatory FK unsatisfied for {mandatory_fks[0].key} at child row {child_row}"
        )

    def _try_resample_child_time_assignment(
        self,
        mandatory_fks: Sequence[ForeignKeySpec],
        child_table: str,
        child_row: int,
        child_time: float,
        table_specs: Mapping[str, TableSpec],
        timestamps: Mapping[str, NDArray[np.float64]],
        row_latents: Mapping[str, NDArray[np.float64]],
        values: dict[str, NDArray[np.float32]],
        state: RouterState,
        peer_rows: dict[str, int],
        snapshot_mode: bool,
        joint: bool,
        fk_stats: dict[str, int],
    ) -> bool:
        parent_times = [
            float(np.max(timestamps[fk.parent_table]))
            if len(timestamps[fk.parent_table]) > 0
            else child_time
            for fk in mandatory_fks
        ]
        new_time = max(parent_times + [child_time])
        candidate_lists = [
            self._candidate_list(
                fk=fk,
                child_row=child_row,
                parent_count=table_specs[fk.parent_table].row_count,
                child_time=new_time,
                parent_timestamps=timestamps[fk.parent_table],
                fanout=state.fanout[fk.key],
                peer_parent_rows=peer_rows,
                snapshot_latest_only=snapshot_mode,
            )
            for fk in mandatory_fks
        ]
        if joint and any(len(candidates) == 0 for candidates in candidate_lists):
            return False
        if not joint and len(candidate_lists[0]) == 0:
            return False
        child_latent = row_latents[child_table][child_row]
        choice = self.joint_sampler.sample_tuple(
            child_row=child_row,
            fks=tuple(mandatory_fks),
            candidate_lists=tuple(candidate_lists),
            child_latent=child_latent,
            row_latents=row_latents,
            child_time=new_time,
            parent_timestamps=timestamps,
            fanout=state.fanout,
            dynamic_popularity=state.dynamic_popularity,
        )
        for fk, parent_row in zip(mandatory_fks, choice):
            timestamps[child_table][child_row] = new_time
            fk_stats["mandatory_fk_timestamp_resample_count"] += 1
            values[fk.key][child_row] = float(parent_row)
            peer_rows[fk.parent_table] = parent_row
            self.updater.register(fk, parent_row, new_time, state)
        return True

    def _try_backoff_assignment(
        self,
        mandatory_fks: Sequence[ForeignKeySpec],
        child_row: int,
        child_time: float,
        table_specs: Mapping[str, TableSpec],
        timestamps: Mapping[str, NDArray[np.float64]],
        row_latents: Mapping[str, NDArray[np.float64]],
        values: dict[str, NDArray[np.float32]],
        state: RouterState,
        peer_rows: dict[str, int],
        snapshot_mode: bool,
        joint: bool,
        fk_stats: dict[str, int],
    ) -> bool:
        fk_stats["mandatory_fk_backoff_count"] += len(mandatory_fks)
        candidate_lists = [
            self._candidate_list(
                fk=fk,
                child_row=child_row,
                parent_count=table_specs[fk.parent_table].row_count,
                child_time=child_time,
                parent_timestamps=timestamps[fk.parent_table],
                fanout=state.fanout[fk.key],
                peer_parent_rows=peer_rows,
                snapshot_latest_only=snapshot_mode,
                relax=True,
            )
            for fk in mandatory_fks
        ]
        if joint and any(len(candidates) == 0 for candidates in candidate_lists):
            return False
        if not joint and len(candidate_lists[0]) == 0:
            return False
        child_latent = row_latents[mandatory_fks[0].child_table][child_row]
        choice = self.joint_sampler.sample_tuple(
            child_row=child_row,
            fks=tuple(mandatory_fks),
            candidate_lists=tuple(candidate_lists),
            child_latent=child_latent,
            row_latents=row_latents,
            child_time=child_time,
            parent_timestamps=timestamps,
            fanout=state.fanout,
            dynamic_popularity=state.dynamic_popularity,
        )
        for fk, parent_row in zip(mandatory_fks, choice):
            values[fk.key][child_row] = float(parent_row)
            peer_rows[fk.parent_table] = parent_row
            self.updater.register(fk, parent_row, child_time, state)
        return True

    def _init_state(
        self,
        table_specs: Mapping[str, TableSpec],
        foreign_keys: Sequence[ForeignKeySpec],
    ) -> RouterState:
        fanout = {
            fk.key: np.zeros(table_specs[fk.parent_table].row_count, dtype=np.int64)
            for fk in foreign_keys
        }
        exposure = {
            table_name: np.zeros(spec.row_count, dtype=np.float64)
            for table_name, spec in table_specs.items()
        }
        recency = {
            table_name: np.zeros(spec.row_count, dtype=np.float64)
            for table_name, spec in table_specs.items()
        }
        dynamic_popularity = {
            table_name: self.rng.lognormal(mean=0.0, sigma=0.5, size=spec.row_count).astype(
                np.float64
            )
            for table_name, spec in table_specs.items()
        }
        return RouterState(
            fanout=fanout,
            exposure=exposure,
            recency=recency,
            dynamic_popularity=dynamic_popularity,
        )


# Backward-compatible alias
ConstraintAwareRelationRouter = TRAPRouter


def _init_mandatory_fk_stats() -> dict[str, int]:
    return {
        "mandatory_fk_total": 0,
        "mandatory_fk_unsatisfied": 0,
        "mandatory_fk_forced_null": 0,
        "mandatory_fk_backoff_count": 0,
        "mandatory_fk_timestamp_resample_count": 0,
        "multi_parent_candidate_empty_count": 0,
        "joint_sampler_backoff_count": 0,
        "joint_sampler_independent_fallback_count": 0,
    }


def _finalize_mandatory_fk_stats(stats: dict[str, int]) -> MandatoryFKStats:
    return MandatoryFKStats(
        mandatory_fk_total=stats["mandatory_fk_total"],
        mandatory_fk_unsatisfied=stats["mandatory_fk_unsatisfied"],
        mandatory_fk_forced_null=stats["mandatory_fk_forced_null"],
        mandatory_fk_backoff_count=stats["mandatory_fk_backoff_count"],
        mandatory_fk_timestamp_resample_count=stats["mandatory_fk_timestamp_resample_count"],
        multi_parent_candidate_empty_count=stats["multi_parent_candidate_empty_count"],
        joint_sampler_backoff_count=stats["joint_sampler_backoff_count"],
        joint_sampler_independent_fallback_count=stats["joint_sampler_independent_fallback_count"],
    )


def _validate_foreign_key_invariants(
    table_specs: Mapping[str, TableSpec],
    foreign_keys: Sequence[ForeignKeySpec],
) -> None:
    for fk in foreign_keys:
        if fk.existence == "mandatory" and fk.nullable:
            raise MandatoryFKSatisfactionError(f"mandatory FK {fk.key} is marked nullable")
        if fk.temporal:
            child = table_specs[fk.child_table]
            parent = table_specs[fk.parent_table]
            if not child.has_timestamp or not parent.has_timestamp:
                raise MandatoryFKSatisfactionError(
                    f"temporal FK {fk.key} references a table without timestamp"
                )


def _child_processing_order(
    table_specs: Mapping[str, TableSpec],
    foreign_keys: Sequence[ForeignKeySpec],
) -> list[str]:
    indegree: dict[str, int] = {name: 0 for name in table_specs}
    children: dict[str, set[str]] = {name: set() for name in table_specs}
    for fk in foreign_keys:
        if fk.parent_table in children and fk.child_table in indegree:
            if fk.child_table not in children[fk.parent_table]:
                children[fk.parent_table].add(fk.child_table)
                indegree[fk.child_table] += 1
    queue = sorted(name for name, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in sorted(children[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(table_specs):
        return sorted(table_specs.keys())
    return order


def _group_multi_parent(
    foreign_keys: Sequence[ForeignKeySpec],
) -> dict[str | None, list[ForeignKeySpec]]:
    groups: dict[str | None, list[ForeignKeySpec]] = {}
    for fk in foreign_keys:
        groups.setdefault(fk.multi_parent_group, []).append(fk)
    return groups


def _softmax_choice(rng: np.random.Generator, scores: NDArray[np.float64]) -> int:
    if len(scores) == 1:
        return 0
    shifted = scores - float(np.max(scores))
    probabilities = np.exp(np.clip(shifted, -50.0, 50.0))
    total = float(np.sum(probabilities))
    if total <= 0.0 or not np.isfinite(total):
        return int(rng.integers(0, len(scores)))
    probabilities = probabilities / total
    return int(rng.choice(len(scores), p=probabilities))
