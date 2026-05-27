from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product

import numpy as np
from numpy.typing import NDArray

from rfm.rdb.config import RDBPriorConfig
from rfm.rdb.types import ForeignKeySpec


class JointParentTupleSampler:
    def __init__(
        self,
        rng: np.random.Generator,
        config: RDBPriorConfig,
        score_fn,
    ) -> None:
        self.rng = rng
        self.config = config
        self.mp_config = config.multi_parent
        self.score_fn = score_fn

    def sample_tuple(
        self,
        child_row: int,
        fks: Sequence[ForeignKeySpec],
        candidate_lists: Sequence[NDArray[np.int64]],
        child_latent: NDArray[np.float64],
        row_latents: Mapping[str, NDArray[np.float64]],
        child_time: float,
        parent_timestamps: Mapping[str, NDArray[np.float64]],
        fanout: Mapping[str, NDArray[np.int64]],
        dynamic_popularity: Mapping[str, NDArray[np.float64]],
        selected_other: Mapping[str, int] | None = None,
    ) -> tuple[int, ...]:
        if len(fks) == 0:
            return ()
        if len(fks) == 1:
            candidates = candidate_lists[0]
            if len(candidates) == 0:
                raise ValueError("cannot sample parent from empty candidate set")
            scores = self.score_fn(
                fk=fks[0],
                candidates=candidates,
                child_latent=child_latent,
                child_time=child_time,
                profile=fks[0].mechanism,
                parent_latents=row_latents[fks[0].parent_table],
                parent_timestamps=parent_timestamps[fks[0].parent_table],
                fanout=fanout[fks[0].key],
                dynamic_popularity=dynamic_popularity[fks[0].parent_table],
                peer_selections=selected_other or {},
                row_latents=row_latents,
            )
            return (int(candidates[_softmax_choice(self.rng, scores)]),)

        if len(fks) <= self.mp_config.max_joint_parents:
            return self._sample_joint_product(
                child_row=child_row,
                fks=fks,
                candidate_lists=candidate_lists,
                child_latent=child_latent,
                row_latents=row_latents,
                child_time=child_time,
                parent_timestamps=parent_timestamps,
                fanout=fanout,
                dynamic_popularity=dynamic_popularity,
            )
        return self._sample_gibbs_block(
            child_row=child_row,
            fks=fks,
            candidate_lists=candidate_lists,
            child_latent=child_latent,
            row_latents=row_latents,
            child_time=child_time,
            parent_timestamps=parent_timestamps,
            fanout=fanout,
            dynamic_popularity=dynamic_popularity,
        )

    def _sample_joint_product(
        self,
        child_row: int,
        fks: Sequence[ForeignKeySpec],
        candidate_lists: Sequence[NDArray[np.int64]],
        child_latent: NDArray[np.float64],
        row_latents: Mapping[str, NDArray[np.float64]],
        child_time: float,
        parent_timestamps: Mapping[str, NDArray[np.float64]],
        fanout: Mapping[str, NDArray[np.int64]],
        dynamic_popularity: Mapping[str, NDArray[np.float64]],
    ) -> tuple[int, ...]:
        trimmed = [
            _trim_candidates(candidates, self.mp_config.candidate_trim, self.rng)
            for candidates in candidate_lists
        ]
        if any(len(c) == 0 for c in trimmed):
            raise ValueError("joint tuple sampling requires non-empty candidate sets")

        tuple_scores: list[float] = []
        tuple_choices: list[tuple[int, ...]] = []
        for combo in product(*[c.tolist() for c in trimmed]):
            choice = tuple(int(v) for v in combo)
            score = self._tuple_energy(
                choice=choice,
                fks=fks,
                child_latent=child_latent,
                row_latents=row_latents,
                child_time=child_time,
                parent_timestamps=parent_timestamps,
                fanout=fanout,
                dynamic_popularity=dynamic_popularity,
            )
            tuple_scores.append(score)
            tuple_choices.append(choice)

        idx = _softmax_choice(self.rng, np.asarray(tuple_scores, dtype=np.float64))
        return tuple_choices[idx]

    def _sample_gibbs_block(
        self,
        child_row: int,
        fks: Sequence[ForeignKeySpec],
        candidate_lists: Sequence[NDArray[np.int64]],
        child_latent: NDArray[np.float64],
        row_latents: Mapping[str, NDArray[np.float64]],
        child_time: float,
        parent_timestamps: Mapping[str, NDArray[np.float64]],
        fanout: Mapping[str, NDArray[np.int64]],
        dynamic_popularity: Mapping[str, NDArray[np.float64]],
    ) -> tuple[int, ...]:
        if any(len(candidates) == 0 for candidates in candidate_lists):
            raise ValueError("Gibbs block sampling requires non-empty candidate sets")
        current = tuple(int(c[0]) for c in candidate_lists)
        for _ in range(self.mp_config.gibbs_iterations):
            for idx, fk in enumerate(fks):
                candidates = _trim_candidates(
                    candidate_lists[idx], self.mp_config.candidate_trim, self.rng
                )
                if len(candidates) == 0:
                    raise ValueError("Gibbs block sampling requires non-empty candidate sets")
                peer = {fks[j].parent_table: current[j] for j in range(len(fks)) if j != idx}
                scores = np.array(
                    [
                        self._tuple_energy(
                            choice=tuple(
                                current[i] if i != idx else int(c) for i in range(len(fks))
                            ),
                            fks=fks,
                            child_latent=child_latent,
                            row_latents=row_latents,
                            child_time=child_time,
                            parent_timestamps=parent_timestamps,
                            fanout=fanout,
                            dynamic_popularity=dynamic_popularity,
                        )
                        for c in candidates
                    ],
                    dtype=np.float64,
                )
                current = tuple(
                    current[i] if i != idx else int(candidates[_softmax_choice(self.rng, scores)])
                    for i in range(len(fks))
                )
        return current

    def _tuple_energy(
        self,
        choice: tuple[int, ...],
        fks: Sequence[ForeignKeySpec],
        child_latent: NDArray[np.float64],
        row_latents: Mapping[str, NDArray[np.float64]],
        child_time: float,
        parent_timestamps: Mapping[str, NDArray[np.float64]],
        fanout: Mapping[str, NDArray[np.int64]],
        dynamic_popularity: Mapping[str, NDArray[np.float64]],
    ) -> float:
        score = 0.0
        peer: dict[str, int] = {}
        for idx, fk in enumerate(fks):
            parent_row = choice[idx]
            candidates = np.array([parent_row], dtype=np.int64)
            peer_selections = {table: row for table, row in peer.items()}
            marginal = self.score_fn(
                fk=fk,
                candidates=candidates,
                child_latent=child_latent,
                child_time=child_time,
                profile=fk.mechanism,
                parent_latents=row_latents[fk.parent_table],
                parent_timestamps=parent_timestamps[fk.parent_table],
                fanout=fanout[fk.key],
                dynamic_popularity=dynamic_popularity[fk.parent_table],
                peer_selections=peer_selections,
                row_latents=row_latents,
            )
            score += float(marginal[0])
            peer[fk.parent_table] = parent_row

        for i in range(len(fks)):
            for j in range(i + 1, len(fks)):
                score += 4.0 * self._pair_compat(
                    fks[i],
                    fks[j],
                    choice[i],
                    choice[j],
                    row_latents,
                )
        return score

    def _pair_compat(
        self,
        fk_i: ForeignKeySpec,
        fk_j: ForeignKeySpec,
        row_i: int,
        row_j: int,
        row_latents: Mapping[str, NDArray[np.float64]],
    ) -> float:
        latent_i = row_latents[fk_i.parent_table][row_i]
        latent_j = row_latents[fk_j.parent_table][row_j]
        mp = self.mp_config
        score = mp.latent_compat_weight * float(latent_i @ latent_j) / np.sqrt(len(latent_i))
        parent_count_i = row_latents[fk_i.parent_table].shape[0]
        parent_count_j = row_latents[fk_j.parent_table].shape[0]
        rank_i = row_i / max(float(parent_count_i - 1), 1.0)
        rank_j = row_j / max(float(parent_count_j - 1), 1.0)
        score -= mp.rank_compat_weight * abs(rank_i - rank_j)

        segment_i = latent_i[0]
        segment_j = latent_j[0]
        same_segment = abs(segment_i - segment_j)
        profile = fk_i.mechanism
        if profile.attachment == "bridge_pairing":
            score += self.mp_config.bridge_pair_energy_weight * (
                profile.bridge_same_segment_bias - same_segment
            )
        else:
            score -= mp.segment_compat_weight * same_segment
        return score


def _trim_candidates(
    candidates: NDArray[np.int64], max_count: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    if len(candidates) <= max_count:
        return candidates
    indices = rng.choice(len(candidates), size=max_count, replace=False)
    return candidates[np.sort(indices)]


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
