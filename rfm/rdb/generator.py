from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from rfm.rdb.config import RDBPriorConfig
from rfm.rdb.content import RelationalSCMGenerator
from rfm.rdb.feasibility import SchemaFeasibilityError, preflight_schema_feasibility
from rfm.rdb.mechanism import MechanismProfileSampler
from rfm.rdb.router import MandatoryFKSatisfactionError, TRAPRouter
from rfm.rdb.schema import SchemaGrammar
from rfm.rdb.types import (
    FeatureColumnType,
    ForeignKeySpec,
    ParentMessage,
    RelationalDataset,
    RelationalPretrainSpec,
    RelationalSplitKind,
    RelationalTargetFamily,
    RelationalTargetSpec,
    RelationalTask,
    RelationMessageSet,
    SchemaGraph,
    TableSpec,
    TargetDependencyKind,
    TargetDifficultyMetrics,
    TargetDifficultyProbeView,
)
from rfm.types import TaskType


class RelationalPriorGenerator:
    def __init__(
        self,
        config: RDBPriorConfig | None = None,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if seed is not None and rng is not None:
            raise ValueError("seed and rng cannot both be provided")
        self.config = config or RDBPriorConfig()
        self.rng = rng or np.random.default_rng(self.config.seed if seed is None else seed)
        self.schema_grammar = SchemaGrammar(self.rng, self.config)
        self.profile_sampler = MechanismProfileSampler(self.rng, self.config)
        self.router = TRAPRouter(self.rng, self.config)
        self.content = RelationalSCMGenerator(self.rng, self.config)
        self.last_task_rejection_reason: str | None = None

    def sample_database(self) -> RelationalDataset:
        last_error: MandatoryFKSatisfactionError | SchemaFeasibilityError | None = None
        for attempt in range(self.config.mandatory_fk_max_retries):
            try:
                database = self._sample_database_once()
                metadata = dict(database.metadata)
                metadata["database_sampling_stats"] = {
                    "attempts": attempt + 1,
                    "retries": attempt,
                    "database_retry_mean": float(attempt),
                    "database_retry_p95": float(attempt),
                }
                return replace(database, metadata=metadata)
            except (MandatoryFKSatisfactionError, SchemaFeasibilityError) as exc:
                last_error = exc
                if attempt + 1 >= self.config.mandatory_fk_max_retries:
                    raise RuntimeError(
                        f"failed to sample database after {self.config.mandatory_fk_max_retries} retries"
                    ) from exc
        if last_error is not None:
            raise RuntimeError("failed to sample database") from last_error
        raise RuntimeError("failed to sample database")

    def _sample_database_once(self) -> RelationalDataset:
        schema = self.schema_grammar.sample()
        profiles = self.profile_sampler.sample_all(schema)
        schema = self.profile_sampler.apply_profiles(schema, profiles)
        row_latents = self._sample_row_latents(schema.tables)
        timestamps = self._sample_timestamps(schema.tables)
        schema, feasibility_report = preflight_schema_feasibility(schema, timestamps)
        routing = self.router.route(schema, timestamps, row_latents)
        effective_timestamps = routing.timestamps
        tables, feature_missing_masks, content_metadata = self._assemble_tables(
            schema=schema,
            timestamps=effective_timestamps,
            row_latents=row_latents,
            routing=routing,
        )
        return RelationalDataset(
            tables=tables,
            table_specs=schema.tables,
            foreign_keys=schema.edges,
            foreign_key_null_masks=routing.null_masks,
            feature_missing_masks=feature_missing_masks,
            row_embeddings={key: value.astype(np.float32) for key, value in row_latents.items()},
            metadata={
                "prior_type": "rdb",
                "mechanism_counts": routing.mechanism_counts,
                "mandatory_fk_stats": routing.mandatory_fk_stats.to_dict(),
                "schema_archetype": schema.archetype,
                "table_roles": {name: spec.role for name, spec in schema.tables.items()},
                "edge_intents": {key: spec.intent for key, spec in schema.edge_intents.items()},
                "edge_metadata": {
                    fk.key: {
                        "semantic": fk.semantic,
                        "cardinality": fk.cardinality,
                        "existence": fk.existence,
                        "nullable": fk.nullable,
                        "temporal": fk.temporal,
                        "capacity": fk.capacity,
                    }
                    for fk in schema.edges
                },
                "feasibility_preflight": feasibility_report.to_dict(),
                "content_metadata": content_metadata,
            },
        )

    def sample_task(self, database: RelationalDataset) -> RelationalTask:
        if self.rng.random() < self.config.explicit_relational_target_probability:
            explicit_task = self._sample_explicit_task(database)
            if explicit_task is not None:
                return explicit_task

        candidates = self._task_candidates(database)
        if len(candidates) == 0:
            raise ValueError("database has no usable target columns")
        content_metadata = database.metadata.get("content_metadata", {})
        if isinstance(content_metadata, dict):
            cross_table_candidates = [
                candidate
                for candidate in candidates
                if self._candidate_uses_parent_features(database, candidate, content_metadata)
            ]
            if (
                cross_table_candidates
                and self.rng.random() < self.config.cross_table_target_probability
            ):
                candidates = cross_table_candidates

        last_error: ValueError | None = None
        for candidate_idx in self.rng.permutation(len(candidates)):
            target_table, target_column, feature_type = candidates[int(candidate_idx)]
            task_type = self._task_type_for_feature(feature_type, self.config.task)
            try:
                split_kind, train_indices, test_indices = self._sample_split(
                    database=database,
                    target_table=target_table,
                    target_column=target_column,
                    task_type=task_type,
                )
            except ValueError as exc:
                last_error = exc
                continue
            num_classes = None
            if task_type == "classification":
                values = database.column_values(target_table, target_column)
                task_indices = np.concatenate([train_indices, test_indices])
                num_classes = int(len(np.unique(values[task_indices].astype(np.int64))))

            dependency_kind, parent_tables, causal_sources = self._target_dependency_metadata(
                database=database,
                target_table=target_table,
                target_column=target_column,
                content_metadata=content_metadata if isinstance(content_metadata, dict) else {},
            )
            return RelationalTask(
                target_table=target_table,
                target_column=target_column,
                task_type=task_type,
                train_indices=train_indices,
                test_indices=test_indices,
                split_kind=split_kind,
                num_classes=num_classes,
                has_cross_table_path=dependency_kind != "focal_only",
                parent_tables_in_path=parent_tables,
                target_dependency_kind=dependency_kind,
                causal_feature_sources=causal_sources,
            )
        if last_error is not None:
            raise ValueError(
                "database has no target column with a valid train/test split"
            ) from last_error
        raise ValueError("database has no usable target columns")

    def sample_pretrain_spec(
        self, database: RelationalDataset | None = None
    ) -> RelationalPretrainSpec:
        database = database or self.sample_database()
        task = self.sample_task(database)
        focal_table = task.target_table
        neighbor_tables, join_paths = self._neighbor_closure(database, focal_table)
        topology_summary = self._topology_summary(database)
        return RelationalPretrainSpec(
            database=database,
            task=task,
            focal_table=focal_table,
            neighbor_tables=neighbor_tables,
            join_paths=join_paths,
            topology_summary=topology_summary,
        )

    def sample_explicit_task(
        self,
        database: RelationalDataset,
        family: RelationalTargetFamily,
        task_type: TaskType,
    ) -> RelationalTask | None:
        self.last_task_rejection_reason = None
        built = self._build_explicit_target(database, family)
        if built is None:
            self.last_task_rejection_reason = "family_unconstructable"
            return None
        target_table, raw_signal, target_spec, causal_sources = built
        task = self._materialize_explicit_task(
            database=database,
            family=family,
            target_table=target_table,
            raw_signal=raw_signal,
            target_spec=target_spec,
            causal_sources=causal_sources,
            task_type=task_type,
        )
        if task is None and self.last_task_rejection_reason is None:
            self.last_task_rejection_reason = "task_unavailable"
        return task

    def _sample_explicit_task(self, database: RelationalDataset) -> RelationalTask | None:
        cross_table = self.rng.random() < self.config.cross_table_target_probability
        primary_families: tuple[RelationalTargetFamily, ...]
        if cross_table:
            primary_families = (
                "parent_feature",
                "parent_child_interaction",
                "multi_parent",
                "topology_driven",
            )
        else:
            primary_families = ("local_only",)
        fallback_families: tuple[RelationalTargetFamily, ...] = (
            "local_only",
            "parent_feature",
            "parent_child_interaction",
            "multi_parent",
            "topology_driven",
        )
        fallback_only_families = tuple(
            family for family in fallback_families if family not in primary_families
        )
        task_types: tuple[TaskType, ...]
        if self.config.task is not None:
            task_types = (self.config.task,)
        elif cross_table:
            task_types = ("regression", "classification")
        else:
            task_types = ("classification", "regression")

        family_order = [
            primary_families[int(family_idx)]
            for family_idx in self.rng.permutation(len(primary_families))
        ]
        family_order.extend(
            fallback_only_families[int(family_idx)]
            for family_idx in self.rng.permutation(len(fallback_only_families))
        )
        for family in family_order:
            built = self._build_explicit_target(database, family)
            if built is None:
                continue
            target_table, raw_signal, target_spec, causal_sources = built
            for task_type in task_types:
                task = self._materialize_explicit_task(
                    database=database,
                    family=family,
                    target_table=target_table,
                    raw_signal=raw_signal,
                    target_spec=target_spec,
                    causal_sources=causal_sources,
                    task_type=task_type,
                )
                if task is not None:
                    return task
        return None

    def _materialize_explicit_task(
        self,
        database: RelationalDataset,
        family: RelationalTargetFamily,
        target_table: str,
        raw_signal: NDArray[np.float64],
        target_spec: RelationalTargetSpec,
        causal_sources: dict[str, tuple[str, ...]],
        task_type: TaskType,
    ) -> RelationalTask | None:
        target_values = self._materialize_target_values(raw_signal, task_type)
        try:
            split_kind, train_indices, test_indices = self._sample_split_from_values(
                database=database,
                target_table=target_table,
                target_values=target_values,
                task_type=task_type,
            )
        except ValueError as exc:
            self.last_task_rejection_reason = _task_rejection_reason(exc, prefix="invalid_split")
            return None
        difficulty_metrics = None
        if self.config.explicit_target_difficulty_filter:
            try:
                difficulty_metrics = self._evaluate_explicit_target_difficulty(
                    database=database,
                    target_table=target_table,
                    target_values=target_values,
                    task_type=task_type,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    family=family,
                )
            except ValueError as exc:
                self.last_task_rejection_reason = _task_rejection_reason(
                    exc, prefix="difficulty_filter"
                )
                return None
        num_classes = None
        if task_type == "classification":
            indices = np.concatenate([train_indices, test_indices])
            num_classes = int(len(np.unique(target_values[indices].astype(np.int64))))
        dependency_kind = _dependency_kind_for_family(family)
        if difficulty_metrics is not None:
            target_spec = replace(target_spec, difficulty_metrics=difficulty_metrics)
        return RelationalTask(
            target_table=target_table,
            target_column=target_spec.target_column,
            task_type=task_type,
            train_indices=train_indices,
            test_indices=test_indices,
            split_kind=split_kind,
            num_classes=num_classes,
            has_cross_table_path=family != "local_only",
            parent_tables_in_path=target_spec.parent_tables_used,
            target_dependency_kind=dependency_kind,
            causal_feature_sources=causal_sources,
            target_spec=target_spec,
            target_values=target_values.astype(np.float32, copy=False),
        )

    def _build_explicit_target(
        self,
        database: RelationalDataset,
        family: RelationalTargetFamily,
    ) -> tuple[str, NDArray[np.float64], RelationalTargetSpec, dict[str, tuple[str, ...]]] | None:
        if family == "local_only":
            return self._build_local_target(database)
        if family == "parent_feature":
            return self._build_parent_feature_target(database)
        if family == "parent_child_interaction":
            return self._build_parent_child_interaction_target(database)
        if family == "multi_parent":
            return self._build_multi_parent_target(database)
        if family == "topology_driven":
            return self._build_topology_target(database)
        raise ValueError(f"unknown relational target family {family}")

    def _build_local_target(
        self,
        database: RelationalDataset,
    ) -> tuple[str, NDArray[np.float64], RelationalTargetSpec, dict[str, tuple[str, ...]]] | None:
        candidates = [
            (table_name, spec)
            for table_name, spec in database.table_specs.items()
            if len(spec.feature_columns) >= 1
        ]
        if not candidates:
            return None
        table_name, spec = candidates[int(self.rng.integers(0, len(candidates)))]
        columns = tuple(
            column.name for column in spec.feature_columns[: min(2, len(spec.feature_columns))]
        )
        signal = np.zeros(spec.row_count, dtype=np.float64)
        for weight, column_name in zip((1.0, 0.5), columns):
            signal += weight * _column_signal(database, table_name, column_name)
        signal = self._add_target_noise(_standardize(signal), signal_to_noise_ratio=4.0)
        target_column = "__relational_target__"
        spec_obj = RelationalTargetSpec(
            focal_table=table_name,
            target_column=target_column,
            target_family="local_only",
            join_path=(table_name,),
            parent_tables_used=(),
            parent_columns_used=(),
            topology_features_used=(),
            causal_formula_type="linear_focal_features",
            signal_to_noise_ratio=4.0,
            focal_only_expected_difficulty=0.35,
        )
        return (
            table_name,
            signal,
            spec_obj,
            {target_column: tuple(f"{table_name}.{name}" for name in columns)},
        )

    def _build_parent_feature_target(
        self,
        database: RelationalDataset,
    ) -> tuple[str, NDArray[np.float64], RelationalTargetSpec, dict[str, tuple[str, ...]]] | None:
        choice = self._sample_joinable_fk(database)
        if choice is None:
            return None
        fk, parent_column = choice
        parent_signal = _joined_parent_signal(database, fk, parent_column)
        signal = self._add_target_noise(parent_signal, signal_to_noise_ratio=3.0)
        target_column = "__relational_target__"
        parent_source = f"{fk.parent_table}.{parent_column}"
        spec_obj = RelationalTargetSpec(
            focal_table=fk.child_table,
            target_column=target_column,
            target_family="parent_feature",
            join_path=(fk.child_table, fk.parent_table),
            parent_tables_used=(fk.parent_table,),
            parent_columns_used=(parent_source,),
            topology_features_used=(),
            causal_formula_type="linear_parent_feature",
            signal_to_noise_ratio=3.0,
            focal_only_expected_difficulty=0.75,
        )
        return fk.child_table, signal, spec_obj, {target_column: (parent_source,)}

    def _build_parent_child_interaction_target(
        self,
        database: RelationalDataset,
    ) -> tuple[str, NDArray[np.float64], RelationalTargetSpec, dict[str, tuple[str, ...]]] | None:
        choice = self._sample_joinable_fk(database)
        if choice is None:
            return None
        fk, parent_column = choice
        focal_columns = database.table_specs[fk.child_table].feature_columns
        if not focal_columns:
            return None
        focal_column = focal_columns[int(self.rng.integers(0, len(focal_columns)))].name
        focal_signal = _column_signal(database, fk.child_table, focal_column)
        parent_signal = _joined_parent_signal(database, fk, parent_column)
        signal = self._add_target_noise(focal_signal * parent_signal, signal_to_noise_ratio=3.0)
        target_column = "__relational_target__"
        focal_source = f"{fk.child_table}.{focal_column}"
        parent_source = f"{fk.parent_table}.{parent_column}"
        spec_obj = RelationalTargetSpec(
            focal_table=fk.child_table,
            target_column=target_column,
            target_family="parent_child_interaction",
            join_path=(fk.child_table, fk.parent_table),
            parent_tables_used=(fk.parent_table,),
            parent_columns_used=(parent_source,),
            topology_features_used=(),
            causal_formula_type="focal_times_parent",
            signal_to_noise_ratio=3.0,
            focal_only_expected_difficulty=0.8,
        )
        return fk.child_table, signal, spec_obj, {target_column: (focal_source, parent_source)}

    def _build_multi_parent_target(
        self,
        database: RelationalDataset,
    ) -> tuple[str, NDArray[np.float64], RelationalTargetSpec, dict[str, tuple[str, ...]]] | None:
        child_tables = sorted({fk.child_table for fk in database.foreign_keys})
        for child_table in child_tables:
            fks = [
                fk
                for fk in database.foreign_keys
                if fk.child_table == child_table
                and len(database.table_specs[fk.parent_table].feature_columns) > 0
            ]
            if len(fks) < 2:
                continue
            selected = tuple(fks[idx] for idx in self.rng.choice(len(fks), size=2, replace=False))
            parent_columns = tuple(
                database.table_specs[fk.parent_table].feature_columns[0].name for fk in selected
            )
            parent_signals = tuple(
                _joined_parent_signal(database, fk, column_name)
                for fk, column_name in zip(selected, parent_columns)
            )
            signal = (
                parent_signals[0] + parent_signals[1] + 0.75 * parent_signals[0] * parent_signals[1]
            )
            signal = self._add_target_noise(signal, signal_to_noise_ratio=3.0)
            target_column = "__relational_target__"
            parent_sources = tuple(
                f"{fk.parent_table}.{column_name}"
                for fk, column_name in zip(selected, parent_columns)
            )
            spec_obj = RelationalTargetSpec(
                focal_table=child_table,
                target_column=target_column,
                target_family="multi_parent",
                join_path=(child_table, *(fk.parent_table for fk in selected)),
                parent_tables_used=tuple(fk.parent_table for fk in selected),
                parent_columns_used=parent_sources,
                topology_features_used=(),
                causal_formula_type="two_parent_sum_and_product",
                signal_to_noise_ratio=3.0,
                focal_only_expected_difficulty=0.85,
            )
            return child_table, signal, spec_obj, {target_column: parent_sources}
        return None

    def _build_topology_target(
        self,
        database: RelationalDataset,
    ) -> tuple[str, NDArray[np.float64], RelationalTargetSpec, dict[str, tuple[str, ...]]] | None:
        incoming = [
            fk
            for fk in database.foreign_keys
            if database.table_specs[fk.parent_table].row_count >= 4
        ]
        if incoming:
            fk = incoming[int(self.rng.integers(0, len(incoming)))]
            signal = _parent_fanout_signal(database, fk)
            target_table = fk.parent_table
            topology_name = f"fanout:{fk.key}"
            join_path = (fk.parent_table, fk.child_table)
        else:
            choice = self._sample_joinable_fk(database)
            if choice is None:
                return None
            fk, _ = choice
            signal = _child_parent_fanout_signal(database, fk)
            target_table = fk.child_table
            topology_name = f"joined_parent_fanout:{fk.key}"
            join_path = (fk.child_table, fk.parent_table)
        if float(np.std(signal)) <= 1e-12:
            return None
        signal = self._add_target_noise(signal, signal_to_noise_ratio=3.0)
        target_column = "__relational_target__"
        spec_obj = RelationalTargetSpec(
            focal_table=target_table,
            target_column=target_column,
            target_family="topology_driven",
            join_path=join_path,
            parent_tables_used=(fk.parent_table,) if target_table == fk.child_table else (),
            parent_columns_used=(),
            topology_features_used=(topology_name,),
            causal_formula_type="fanout_state",
            signal_to_noise_ratio=3.0,
            focal_only_expected_difficulty=0.9,
        )
        return target_table, signal, spec_obj, {target_column: (topology_name,)}

    def _sample_joinable_fk(self, database: RelationalDataset) -> tuple[ForeignKeySpec, str] | None:
        candidates: list[tuple[ForeignKeySpec, str]] = []
        for fk in database.foreign_keys:
            parent_columns = database.table_specs[fk.parent_table].feature_columns
            if len(parent_columns) == 0:
                continue
            values = database.column_values(fk.child_table, fk.child_column)
            if int(np.isfinite(values).sum()) < max(4, int(0.25 * len(values))):
                continue
            column = parent_columns[int(self.rng.integers(0, len(parent_columns)))]
            candidates.append((fk, column.name))
        if not candidates:
            return None
        return candidates[int(self.rng.integers(0, len(candidates)))]

    def _add_target_noise(
        self,
        signal: NDArray[np.float64],
        signal_to_noise_ratio: float,
    ) -> NDArray[np.float64]:
        standardized = _standardize(signal.astype(np.float64, copy=False))
        noise_scale = 1.0 / max(signal_to_noise_ratio, 1e-6)
        noisy = standardized + self.rng.normal(0.0, noise_scale, size=len(standardized))
        return _standardize(noisy)

    def _materialize_target_values(
        self,
        signal: NDArray[np.float64],
        task_type: TaskType,
    ) -> NDArray[np.float32]:
        if task_type == "regression":
            return _standardize(signal).astype(np.float32)
        return _quantile_classes(signal, class_count=3).astype(np.float32)

    def _evaluate_explicit_target_difficulty(
        self,
        database: RelationalDataset,
        target_table: str,
        target_values: NDArray[np.float32],
        task_type: TaskType,
        train_indices: NDArray[np.int64],
        test_indices: NDArray[np.int64],
        family: RelationalTargetFamily,
    ) -> TargetDifficultyMetrics:
        focal_score = _score_target_difficulty(
            features=_focal_feature_matrix(database, target_table),
            target_values=target_values,
            task_type=task_type,
            train_indices=train_indices,
            test_indices=test_indices,
        )
        joined_score = _score_target_difficulty(
            features=_joined_feature_matrix(database, target_table),
            target_values=target_values,
            task_type=task_type,
            train_indices=train_indices,
            test_indices=test_indices,
        )
        topology_score = _score_target_difficulty(
            features=_topology_feature_matrix(database, target_table),
            target_values=target_values,
            task_type=task_type,
            train_indices=train_indices,
            test_indices=test_indices,
        )
        accepted_view = _difficulty_view_for_family(family)
        accepted_score = {
            "focal_only": focal_score,
            "joined_flat": joined_score,
            "topology": topology_score,
        }[accepted_view]

        baseline_score: float | None = None
        class_entropy: float | None = None
        min_class_fraction: float | None = None
        if task_type == "classification":
            labels = target_values.astype(np.int64, copy=False)
            task_indices = np.concatenate([train_indices, test_indices])
            class_entropy = _normalized_class_entropy(labels[task_indices])
            min_class_fraction = _min_class_fraction(labels[task_indices])
            baseline_score = _majority_score(labels[test_indices])
            if class_entropy < self.config.explicit_target_min_class_entropy:
                raise ValueError("explicit target difficulty filter rejected low class entropy")
            if min_class_fraction < self.config.explicit_target_min_class_fraction:
                raise ValueError("explicit target difficulty filter rejected class imbalance")
            if accepted_score > self.config.explicit_target_max_classification_probe_accuracy:
                raise ValueError(
                    "explicit target difficulty filter rejected near-perfect classification target"
                )
            if (
                accepted_score
                < baseline_score + self.config.explicit_target_min_classification_probe_margin
            ):
                raise ValueError(
                    "explicit target difficulty filter rejected low-signal classification target"
                )
        else:
            if accepted_score > self.config.explicit_target_max_regression_probe_r2:
                raise ValueError(
                    "explicit target difficulty filter rejected near-perfect regression target"
                )
            if accepted_score < self.config.explicit_target_min_regression_probe_r2:
                raise ValueError(
                    "explicit target difficulty filter rejected low-signal regression target"
                )

        relational_gain = _relational_probe_gain(
            family=family,
            focal_score=focal_score,
            joined_score=joined_score,
            topology_score=topology_score,
        )
        if (
            relational_gain is not None
            and relational_gain < self.config.explicit_target_min_relational_probe_gain
        ):
            raise ValueError("explicit target difficulty filter rejected weak relational gain")

        return TargetDifficultyMetrics(
            accepted_view=accepted_view,
            accepted_score=accepted_score,
            focal_score=focal_score,
            joined_score=joined_score,
            topology_score=topology_score,
            baseline_score=baseline_score,
            class_entropy=class_entropy,
            min_class_fraction=min_class_fraction,
            relational_gain=relational_gain,
        )

    def _sample_row_latents(
        self, table_specs: Mapping[str, TableSpec]
    ) -> dict[str, NDArray[np.float64]]:
        role_vectors = {
            role: self.rng.normal(0.0, 0.7, size=self.config.latent_dim)
            for role in self.config.table_roles
        }
        row_latents: dict[str, NDArray[np.float64]] = {}
        for table_name, spec in table_specs.items():
            segment_count = max(2, min(12, int(np.sqrt(spec.row_count))))
            segment_ids = self.rng.integers(0, segment_count, size=spec.row_count)
            segment_centers = self.rng.normal(
                0.0, 0.9, size=(segment_count, self.config.latent_dim)
            )
            latent = role_vectors[spec.role] + segment_centers[segment_ids]
            latent = latent + self.rng.normal(0.0, 0.25, size=latent.shape)
            latent[:, 0] = _standardize(
                segment_ids.astype(np.float64) + self.rng.normal(0.0, 0.05, size=spec.row_count)
            )
            row_latents[table_name] = latent.astype(np.float64)
        return row_latents

    def _sample_timestamps(
        self, table_specs: Mapping[str, TableSpec]
    ) -> dict[str, NDArray[np.float64]]:
        timestamps: dict[str, NDArray[np.float64]] = {}
        for table_name, spec in table_specs.items():
            start, stop = _role_time_range(spec.role)
            base = np.linspace(start, stop, spec.row_count)
            jitter = self.rng.normal(0.0, 0.015, size=spec.row_count)
            timestamps[table_name] = np.sort(np.clip(base + jitter, 0.0, 1.0)).astype(np.float64)
        return timestamps

    def _assemble_tables(
        self,
        schema: SchemaGraph,
        timestamps: Mapping[str, NDArray[np.float64]],
        row_latents: Mapping[str, NDArray[np.float64]],
        routing,
    ) -> tuple[dict[str, NDArray[np.float32]], dict[str, NDArray[np.bool_]], dict[str, object]]:
        tables: dict[str, NDArray[np.float32]] = {}
        feature_missing_masks: dict[str, NDArray[np.bool_]] = {}
        content_metadata: dict[str, object] = {}
        foreign_keys = schema.edges

        for table_name in schema.topological_order:
            spec = schema.tables[table_name]
            relation_messages = self._build_relation_messages(
                table_name=table_name,
                row_count=spec.row_count,
                foreign_keys=foreign_keys,
                routing=routing,
                row_latents=row_latents,
                tables=tables,
                table_specs=schema.tables,
            )
            parent_context = self._parent_context(
                table_name, spec.row_count, foreign_keys, routing, row_latents
            )
            topology_context = self._topology_context(
                table_name, spec.row_count, foreign_keys, routing
            )
            row_context = np.concatenate(
                [row_latents[table_name], timestamps[table_name][:, None]], axis=1
            )
            content = self.content.sample_table_columns(
                table=spec,
                row_context=row_context,
                parent_context=parent_context,
                topology_context=topology_context,
                relation_messages=relation_messages,
            )
            table = np.full((spec.row_count, len(spec.columns)), np.nan, dtype=np.float32)
            for column_idx, column in enumerate(spec.columns):
                if column.kind == "primary_key":
                    table[:, column_idx] = np.arange(spec.row_count, dtype=np.float32)
                elif column.kind == "timestamp":
                    table[:, column_idx] = timestamps[table_name].astype(np.float32)
                elif column.kind == "foreign_key":
                    table[:, column_idx] = routing.values[f"{table_name}.{column.name}"]
                elif column.kind == "feature":
                    table[:, column_idx] = content.values[column.name]
            tables[table_name] = table
            feature_missing_masks[table_name] = content.missing_mask
            content_metadata[table_name] = {
                "feature_parent_sources": content.feature_parent_sources,
                "uses_parent_features": content.uses_parent_features,
                "num_exogenous": content.num_exogenous,
                "num_edge_messages": 0
                if relation_messages is None
                else relation_messages.edge_messages.shape[1],
            }
        return tables, feature_missing_masks, content_metadata

    def _build_relation_messages(
        self,
        table_name: str,
        row_count: int,
        foreign_keys: Sequence[ForeignKeySpec],
        routing,
        row_latents: Mapping[str, NDArray[np.float64]],
        tables: Mapping[str, NDArray[np.float32]],
        table_specs: Mapping[str, TableSpec],
    ) -> RelationMessageSet | None:
        child_fks = [fk for fk in foreign_keys if fk.child_table == table_name]
        if len(child_fks) == 0:
            return None

        feature_dim = self.config.parent_feature_message_dim
        edge_profile_dim = 6
        message_dim = self.config.latent_dim + feature_dim + edge_profile_dim + 1
        edge_messages = np.zeros((row_count, len(child_fks) * message_dim), dtype=np.float64)
        messages: list[ParentMessage] = []

        for edge_idx, fk in enumerate(child_fks):
            values = routing.values[fk.key]
            parent_spec = table_specs[fk.parent_table]
            parent_feature_indices = self._select_parent_feature_indices(parent_spec, feature_dim)
            offset = edge_idx * message_dim
            template_message: ParentMessage | None = None

            for row in range(row_count):
                value = values[row]
                is_null = not np.isfinite(value)
                if is_null:
                    continue

                parent_row = int(value)
                parent_latent = row_latents[fk.parent_table][parent_row]
                parent_features = self._extract_parent_features(
                    tables=tables,
                    parent_spec=parent_spec,
                    parent_row=parent_row,
                    feature_indices=parent_feature_indices,
                    feature_dim=feature_dim,
                )
                fanout_state = float(routing.fanout[fk.key][parent_row])
                edge_profile = np.asarray(
                    [
                        fk.mechanism.hub_strength,
                        fk.mechanism.locality_strength,
                        fk.mechanism.temporal_strength,
                        fk.mechanism.compat_strength,
                        float(fk.existence == "optional"),
                        float(fk.temporal),
                    ],
                    dtype=np.float64,
                )
                block = np.concatenate(
                    [
                        parent_latent,
                        parent_features,
                        edge_profile,
                        np.asarray([fanout_state], dtype=np.float64),
                    ]
                )
                edge_messages[row, offset : offset + message_dim] = block
                if template_message is None:
                    template_message = ParentMessage(
                        fk_key=fk.key,
                        parent_table=fk.parent_table,
                        parent_latent=parent_latent,
                        parent_features=parent_features,
                        edge_profile=edge_profile,
                        fanout_state=fanout_state,
                        is_null=False,
                    )

            if template_message is None:
                template_message = ParentMessage(
                    fk_key=fk.key,
                    parent_table=fk.parent_table,
                    parent_latent=np.zeros(self.config.latent_dim, dtype=np.float64),
                    parent_features=np.zeros(feature_dim, dtype=np.float64),
                    edge_profile=np.zeros(edge_profile_dim, dtype=np.float64),
                    fanout_state=0.0,
                    is_null=True,
                )
            messages.append(template_message)

        return RelationMessageSet(messages=tuple(messages), edge_messages=edge_messages)

    def _select_parent_feature_indices(
        self, parent_spec: TableSpec, feature_dim: int
    ) -> tuple[int, ...]:
        feature_columns = parent_spec.feature_columns
        if len(feature_columns) == 0 or feature_dim <= 0:
            return ()
        count = min(feature_dim, len(feature_columns))
        if count == len(feature_columns):
            return tuple(range(count))
        indices = self.rng.choice(len(feature_columns), size=count, replace=False)
        return tuple(sorted(int(idx) for idx in indices))

    def _extract_parent_features(
        self,
        tables: Mapping[str, NDArray[np.float32]],
        parent_spec: TableSpec,
        parent_row: int,
        feature_indices: tuple[int, ...],
        feature_dim: int,
    ) -> NDArray[np.float64]:
        features = np.zeros(feature_dim, dtype=np.float64)
        if parent_spec.name not in tables or len(feature_indices) == 0:
            return features
        parent_table = tables[parent_spec.name]
        for out_idx, feature_idx in enumerate(feature_indices):
            column = parent_spec.feature_columns[feature_idx]
            column_idx = parent_spec.column_index(column.name)
            value = float(parent_table[parent_row, column_idx])
            features[out_idx] = 0.0 if not np.isfinite(value) else value
        return features

    def _candidate_uses_parent_features(
        self,
        database: RelationalDataset,
        candidate: tuple[str, str, FeatureColumnType],
        content_metadata: dict[str, object],
    ) -> bool:
        target_table, target_column, _ = candidate
        table_meta = content_metadata.get(target_table, {})
        if not isinstance(table_meta, dict):
            return False
        uses_parent_features = table_meta.get("uses_parent_features", {})
        if not isinstance(uses_parent_features, dict):
            return False
        return bool(uses_parent_features.get(target_column, False))

    def _target_dependency_metadata(
        self,
        database: RelationalDataset,
        target_table: str,
        target_column: str,
        content_metadata: dict[str, object],
    ) -> tuple[TargetDependencyKind, tuple[str, ...], dict[str, tuple[str, ...]] | None]:
        table_meta = content_metadata.get(target_table, {})
        if not isinstance(table_meta, dict):
            return "focal_only", (), None
        feature_parent_sources = table_meta.get("feature_parent_sources", {})
        uses_parent_features = table_meta.get("uses_parent_features", {})
        if not isinstance(feature_parent_sources, dict) or not isinstance(
            uses_parent_features, dict
        ):
            return "focal_only", (), None

        parent_tables = feature_parent_sources.get(target_column, ())
        if not isinstance(parent_tables, tuple):
            parent_tables = tuple(parent_tables) if parent_tables else ()
        uses_parent = bool(uses_parent_features.get(target_column, False))
        if not uses_parent or len(parent_tables) == 0:
            return (
                "focal_only",
                (),
                dict(feature_parent_sources) if feature_parent_sources else None,
            )

        dependency_kind: TargetDependencyKind = "parent_feature"
        if any(
            fk.parent_table in parent_tables
            for fk in database.foreign_keys
            if fk.child_table == target_table
        ):
            dependency_kind = "joined"
        return dependency_kind, parent_tables, dict(feature_parent_sources)

    def _parent_context(
        self,
        table_name: str,
        row_count: int,
        foreign_keys: Sequence[ForeignKeySpec],
        routing,
        row_latents: Mapping[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        context = np.zeros((row_count, self.config.latent_dim), dtype=np.float64)
        counts = np.zeros(row_count, dtype=np.float64)
        for fk in foreign_keys:
            if fk.child_table != table_name:
                continue
            values = routing.values[fk.key]
            mask = np.isfinite(values)
            parent_rows = values[mask].astype(np.int64)
            context[mask] += row_latents[fk.parent_table][parent_rows]
            counts[mask] += 1.0
        nonzero = counts > 0.0
        context[nonzero] = context[nonzero] / counts[nonzero, None]
        return context

    def _topology_context(
        self,
        table_name: str,
        row_count: int,
        foreign_keys: Sequence[ForeignKeySpec],
        routing,
    ) -> NDArray[np.float64]:
        dim = self.config.topology_context_dim
        topology = np.zeros((row_count, dim), dtype=np.float64)
        for fk in foreign_keys:
            if fk.parent_table == table_name:
                fanout = routing.fanout[fk.key].astype(np.float64)
                _add_topology_feature(topology, 0, fanout)
                _add_topology_feature(topology, 1, np.log1p(fanout))
            if fk.child_table == table_name:
                values = routing.values[fk.key]
                _add_topology_feature(topology, 2, np.isfinite(values).astype(np.float64))
                _add_topology_feature(topology, 3, fk.mechanism.hub_strength)
                _add_topology_feature(topology, 4, fk.mechanism.locality_strength)
                _add_topology_feature(topology, 5, fk.mechanism.temporal_strength)
        for col in range(topology.shape[1]):
            if float(np.std(topology[:, col])) > 1e-12:
                topology[:, col] = _standardize(topology[:, col])
        return topology

    def _neighbor_closure(
        self,
        database: RelationalDataset,
        focal_table: str,
    ) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
        neighbors: set[str] = set()
        paths: list[tuple[str, ...]] = []
        for fk in database.foreign_keys:
            if fk.child_table == focal_table:
                neighbors.add(fk.parent_table)
                paths.append((focal_table, fk.parent_table))
            elif fk.parent_table == focal_table:
                neighbors.add(fk.child_table)
                paths.append((focal_table, fk.child_table))
        return tuple(sorted(neighbors)), tuple(paths)

    def _topology_summary(self, database: RelationalDataset) -> dict[str, object]:
        attachment_counts: dict[str, int] = {}
        existence_counts: dict[str, int] = {}
        for fk in database.foreign_keys:
            attachment_counts[fk.mechanism.attachment] = (
                attachment_counts.get(fk.mechanism.attachment, 0) + 1
            )
            existence_counts[fk.existence] = existence_counts.get(fk.existence, 0) + 1
        return {
            "attachment_counts": attachment_counts,
            "existence_counts": existence_counts,
            "num_tables": len(database.table_specs),
            "num_foreign_keys": len(database.foreign_keys),
        }

    def _task_candidates(
        self, database: RelationalDataset
    ) -> list[tuple[str, str, FeatureColumnType]]:
        candidates: list[tuple[str, str, FeatureColumnType]] = []
        fallback: list[tuple[str, str, FeatureColumnType]] = []
        for table_name, spec in database.table_specs.items():
            for column in spec.feature_columns:
                values = database.column_values(table_name, column.name)
                finite = values[np.isfinite(values)]
                if len(finite) >= 2 and len(np.unique(finite)) >= 2:
                    candidate = (table_name, column.name, column.value_type)
                    fallback.append(candidate)
                    if self.config.task == "classification" and column.value_type in (
                        "continuous",
                        "count",
                    ):
                        continue
                    if self.config.task == "regression" and column.value_type not in (
                        "continuous",
                        "count",
                    ):
                        continue
                    candidates.append(candidate)
        return candidates if candidates else fallback

    def _task_type_for_feature(
        self, feature_type: FeatureColumnType, forced: TaskType | None
    ) -> TaskType:
        if forced is not None:
            return forced
        if feature_type in ("continuous", "count"):
            return "regression"
        return "classification"

    def _sample_split(
        self,
        database: RelationalDataset,
        target_table: str,
        target_column: str,
        task_type: TaskType,
    ) -> tuple[RelationalSplitKind, NDArray[np.int64], NDArray[np.int64]]:
        target_values = database.column_values(target_table, target_column)
        return self._sample_split_from_values(
            database=database,
            target_table=target_table,
            target_values=target_values,
            task_type=task_type,
        )

    def _sample_split_from_values(
        self,
        database: RelationalDataset,
        target_table: str,
        target_values: NDArray[np.float32],
        task_type: TaskType,
    ) -> tuple[RelationalSplitKind, NDArray[np.int64], NDArray[np.int64]]:
        eligible = np.flatnonzero(np.isfinite(target_values)).astype(np.int64)
        if len(eligible) < 2:
            raise ValueError("target column has fewer than two finite rows")
        train_size = min(max(int(round(0.7 * len(eligible))), 1), len(eligible) - 1)
        timestamp_column = database.table_specs[target_table].timestamp_column
        if (
            self.rng.random() < self.config.temporal_task_probability
            and timestamp_column is not None
        ):
            timestamps = database.column_values(target_table, timestamp_column)[eligible]
            order = eligible[np.argsort(timestamps, kind="mergesort")]
            train_indices = order[:train_size].astype(np.int64)
            test_indices = order[train_size:].astype(np.int64)
            if task_type != "classification" or _classification_split_is_valid(
                target_values,
                train_indices,
                test_indices,
            ):
                return "temporal", train_indices, test_indices
        if task_type != "classification" and self.rng.random() < self.config.ood_task_probability:
            filled = target_values[eligible].astype(np.float64, copy=True)
            order = np.argsort(filled, kind="mergesort")
            ordered_indices = eligible[order]
            return (
                "ood",
                ordered_indices[:train_size].astype(np.int64),
                ordered_indices[train_size:].astype(np.int64),
            )
        if task_type == "classification":
            return "random", *self._sample_stratified_classification_split(target_values, eligible)
        order = self.rng.permutation(eligible).astype(np.int64)
        return "random", order[:train_size], order[train_size:]

    def _sample_stratified_classification_split(
        self,
        target_values: NDArray[np.float32],
        eligible: NDArray[np.int64],
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        labels = target_values[eligible].astype(np.int64)
        classes, counts = np.unique(labels, return_counts=True)
        splittable_classes = classes[counts >= 2]
        if len(splittable_classes) < 2:
            raise ValueError("classification target has fewer than two splittable classes")

        train_parts: list[NDArray[np.int64]] = []
        test_parts: list[NDArray[np.int64]] = []
        for class_id in splittable_classes:
            class_indices = eligible[labels == class_id]
            class_indices = self.rng.permutation(class_indices).astype(np.int64)
            class_train_size = min(
                max(int(round(0.7 * len(class_indices))), 1), len(class_indices) - 1
            )
            train_parts.append(class_indices[:class_train_size])
            test_parts.append(class_indices[class_train_size:])

        train_indices = self.rng.permutation(np.concatenate(train_parts)).astype(np.int64)
        test_indices = self.rng.permutation(np.concatenate(test_parts)).astype(np.int64)
        return train_indices, test_indices


def _role_time_range(role: str) -> tuple[float, float]:
    if role == "dimension/lookup":
        return 0.0, 0.40
    if role == "entity":
        return 0.0, 0.60
    if role == "bridge":
        return 0.15, 0.90
    if role == "activity/event":
        return 0.25, 1.00
    return 0.40, 1.00


def _add_topology_feature(
    topology: NDArray[np.float64],
    column: int,
    values: NDArray[np.float64] | float,
) -> None:
    if column < topology.shape[1]:
        topology[:, column] += values


def _classification_split_is_valid(
    values: NDArray[np.float32],
    train_indices: NDArray[np.int64],
    test_indices: NDArray[np.int64],
) -> bool:
    train_classes = set(np.unique(values[train_indices].astype(np.int64)).tolist())
    test_classes = set(np.unique(values[test_indices].astype(np.int64)).tolist())
    return train_classes == test_classes and len(train_classes) >= 2


def _task_rejection_reason(exc: ValueError, prefix: str) -> str:
    text = str(exc)
    reasons = {
        "low class entropy": "low_class_entropy",
        "class imbalance": "class_imbalance",
        "near-perfect classification": "near_perfect_classification",
        "low-signal classification": "low_signal_classification",
        "near-perfect regression": "near_perfect_regression",
        "low-signal regression": "low_signal_regression",
        "weak relational gain": "weak_relational_gain",
        "fewer than two finite rows": "insufficient_finite_rows",
        "fewer than two splittable classes": "unsplittable_classes",
    }
    for fragment, reason in reasons.items():
        if fragment in text:
            return f"{prefix}:{reason}"
    return f"{prefix}:other"


def _difficulty_view_for_family(family: RelationalTargetFamily) -> TargetDifficultyProbeView:
    if family == "local_only":
        return "focal_only"
    if family == "topology_driven":
        return "topology"
    return "joined_flat"


def _relational_probe_gain(
    family: RelationalTargetFamily,
    focal_score: float,
    joined_score: float,
    topology_score: float,
) -> float | None:
    if family in ("parent_feature", "parent_child_interaction", "multi_parent"):
        return joined_score - focal_score
    if family == "topology_driven":
        return topology_score - focal_score
    return None


def _focal_feature_matrix(
    database: RelationalDataset,
    target_table: str,
) -> NDArray[np.float64]:
    spec = database.table_specs[target_table]
    feature_indices = [spec.column_index(column.name) for column in spec.feature_columns]
    if not feature_indices:
        if spec.timestamp_column is None:
            raise ValueError(f"table {target_table!r} has no focal features")
        feature_indices = [spec.column_index(spec.timestamp_column)]
    return database.tables[target_table][:, feature_indices].astype(np.float64, copy=True)


def _joined_feature_matrix(
    database: RelationalDataset,
    target_table: str,
) -> NDArray[np.float64]:
    blocks = [_focal_feature_matrix(database, target_table)]
    row_count = database.table_specs[target_table].row_count
    for fk in database.foreign_keys:
        if fk.child_table != target_table:
            continue
        parent_spec = database.table_specs[fk.parent_table]
        parent_feature_indices = [
            parent_spec.column_index(column.name) for column in parent_spec.feature_columns
        ]
        if not parent_feature_indices:
            continue
        parent_table = database.tables[fk.parent_table]
        fk_values = database.column_values(target_table, fk.child_column)
        joined = np.full((row_count, len(parent_feature_indices)), np.nan, dtype=np.float64)
        mask = np.isfinite(fk_values)
        if mask.any():
            joined[mask] = parent_table[fk_values[mask].astype(np.int64)][:, parent_feature_indices]
        blocks.append(joined)
    return np.concatenate(blocks, axis=1)


def _topology_feature_matrix(
    database: RelationalDataset,
    target_table: str,
) -> NDArray[np.float64]:
    features: list[NDArray[np.float64]] = []
    row_count = database.table_specs[target_table].row_count
    for fk in database.foreign_keys:
        if fk.child_table == target_table:
            fk_values = database.column_values(fk.child_table, fk.child_column)
            parent_counts = _edge_parent_counts(database, fk)
            joined_fanout = np.zeros(row_count, dtype=np.float64)
            mask = np.isfinite(fk_values)
            if mask.any():
                joined_fanout[mask] = np.log1p(parent_counts[fk_values[mask].astype(np.int64)])
            features.append(np.isfinite(fk_values).astype(np.float64)[:, None])
            features.append(_standardize(joined_fanout)[:, None])
        if fk.parent_table == target_table:
            parent_counts = _edge_parent_counts(database, fk)
            features.append(_standardize(np.log1p(parent_counts))[:, None])
    if not features:
        return np.zeros((row_count, 1), dtype=np.float64)
    return np.concatenate(features, axis=1)


def _edge_parent_counts(database: RelationalDataset, fk: ForeignKeySpec) -> NDArray[np.float64]:
    values = database.column_values(fk.child_table, fk.child_column)
    parent_count = database.table_specs[fk.parent_table].row_count
    parent_rows = values[np.isfinite(values)].astype(np.int64)
    return np.bincount(parent_rows, minlength=parent_count).astype(np.float64)


def _score_target_difficulty(
    features: NDArray[np.float64],
    target_values: NDArray[np.float32],
    task_type: TaskType,
    train_indices: NDArray[np.int64],
    test_indices: NDArray[np.int64],
) -> float:
    prepared = _prepare_difficulty_features(features, train_indices)
    x_train = prepared[train_indices]
    x_test = prepared[test_indices]
    if task_type == "classification":
        labels = target_values.astype(np.int64, copy=False)
        return _ridge_classification_accuracy(
            x_train=x_train,
            y_train=labels[train_indices],
            x_test=x_test,
            y_test=labels[test_indices],
        )
    targets = target_values.astype(np.float64, copy=False)
    return _ridge_regression_r2(
        x_train=x_train,
        y_train=targets[train_indices],
        x_test=x_test,
        y_test=targets[test_indices],
    )


def _prepare_difficulty_features(
    features: NDArray[np.float64],
    train_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    if features.ndim != 2:
        raise ValueError("difficulty features must be a matrix")
    matrix = features.astype(np.float64, copy=True)
    if matrix.shape[1] == 0:
        matrix = np.zeros((matrix.shape[0], 1), dtype=np.float64)
    train = matrix[train_indices]
    means = np.zeros(matrix.shape[1], dtype=np.float64)
    for col_idx in range(matrix.shape[1]):
        finite = np.isfinite(train[:, col_idx])
        if finite.any():
            means[col_idx] = float(np.mean(train[finite, col_idx]))
        missing = ~np.isfinite(matrix[:, col_idx])
        matrix[missing, col_idx] = means[col_idx]
    scales = np.std(matrix[train_indices], axis=0)
    scales[(scales <= 1e-12) | ~np.isfinite(scales)] = 1.0
    return (matrix - means) / scales


def _ridge_classification_accuracy(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    x_test: NDArray[np.float64],
    y_test: NDArray[np.int64],
) -> float:
    classes = np.unique(y_train)
    if len(classes) < 2:
        return _majority_score(y_test)
    targets = (y_train[:, None] == classes[None, :]).astype(np.float64)
    weights = _ridge_weights(_with_bias(x_train), targets, alpha=1.0)
    logits = _with_bias(x_test) @ weights
    predictions = classes[np.argmax(logits, axis=1)]
    return float(np.mean(predictions == y_test))


def _ridge_regression_r2(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
) -> float:
    weights = _ridge_weights(_with_bias(x_train), y_train[:, None], alpha=1.0)
    predictions = (_with_bias(x_test) @ weights).ravel()
    variance = float(np.sum((y_test - float(np.mean(y_test))) ** 2))
    if variance <= 1e-12 or not np.isfinite(variance):
        return 0.0
    residual = float(np.sum((predictions - y_test) ** 2))
    return 1.0 - residual / variance


def _ridge_weights(
    x_train: NDArray[np.float64],
    targets: NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64]:
    regularizer = np.eye(x_train.shape[1], dtype=np.float64) * alpha
    regularizer[0, 0] = 0.0
    gram = x_train.T @ x_train
    rhs = x_train.T @ targets
    try:
        return np.linalg.solve(gram + regularizer, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram + regularizer) @ rhs


def _with_bias(features: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.concatenate([np.ones((features.shape[0], 1), dtype=np.float64), features], axis=1)


def _majority_score(labels: NDArray[np.int64]) -> float:
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    return float(np.max(counts) / len(labels))


def _normalized_class_entropy(labels: NDArray[np.int64]) -> float:
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) < 2:
        return 0.0
    probabilities = counts.astype(np.float64) / float(np.sum(counts))
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(len(counts)))


def _min_class_fraction(labels: NDArray[np.int64]) -> float:
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    return float(np.min(counts) / len(labels))


def _dependency_kind_for_family(family: RelationalTargetFamily) -> TargetDependencyKind:
    if family == "local_only":
        return "focal_only"
    if family == "parent_feature":
        return "parent_feature"
    if family == "parent_child_interaction":
        return "parent_child_interaction"
    if family == "multi_parent":
        return "multi_parent"
    if family == "topology_driven":
        return "topology_driven"
    raise ValueError(f"unknown target family {family}")


def _column_signal(
    database: RelationalDataset,
    table_name: str,
    column_name: str,
) -> NDArray[np.float64]:
    values = database.column_values(table_name, column_name).astype(np.float64, copy=True)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(len(values), dtype=np.float64)
    fill = float(np.mean(values[finite]))
    values[~finite] = fill
    return _standardize(values)


def _joined_parent_signal(
    database: RelationalDataset,
    fk: ForeignKeySpec,
    parent_column: str,
) -> NDArray[np.float64]:
    parent_signal = _column_signal(database, fk.parent_table, parent_column)
    values = database.column_values(fk.child_table, fk.child_column)
    signal = np.zeros(len(values), dtype=np.float64)
    mask = np.isfinite(values)
    if mask.any():
        signal[mask] = parent_signal[values[mask].astype(np.int64)]
    return _standardize(signal)


def _parent_fanout_signal(database: RelationalDataset, fk: ForeignKeySpec) -> NDArray[np.float64]:
    values = database.column_values(fk.child_table, fk.child_column)
    parent_count = database.table_specs[fk.parent_table].row_count
    parent_rows = values[np.isfinite(values)].astype(np.int64)
    counts = np.bincount(parent_rows, minlength=parent_count).astype(np.float64)
    return _standardize(np.log1p(counts))


def _child_parent_fanout_signal(
    database: RelationalDataset, fk: ForeignKeySpec
) -> NDArray[np.float64]:
    parent_fanout = _parent_fanout_signal(database, fk)
    values = database.column_values(fk.child_table, fk.child_column)
    signal = np.zeros(len(values), dtype=np.float64)
    mask = np.isfinite(values)
    if mask.any():
        signal[mask] = parent_fanout[values[mask].astype(np.int64)]
    return _standardize(signal)


def _quantile_classes(values: NDArray[np.float64], class_count: int) -> NDArray[np.int64]:
    ranks = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    labels = np.floor(ranks * class_count / len(values)).astype(np.int64)
    return np.minimum(labels, class_count - 1)


def _standardize(values: NDArray[np.float64]) -> NDArray[np.float64]:
    std = float(np.std(values))
    if std <= 1e-12 or not np.isfinite(std):
        return values - float(np.mean(values))
    return (values - float(np.mean(values))) / std
