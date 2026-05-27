from __future__ import annotations

import unittest

import numpy as np
from numpy.typing import NDArray

from rfm.rdb import (
    RDBPriorConfig,
    RelationalPriorGenerator,
    RoleGrammarConfig,
    SchemaArchetypeConfig,
)
from rfm.rdb.multiparent import JointParentTupleSampler
from rfm.rdb.router import CandidateConstraintGate, RelationFieldScorer
from rfm.rdb.types import ForeignKeySpec, MechanismProfile


class RelationalPriorGeneratorTest(unittest.TestCase):
    def test_sample_database_is_reproducible_for_fixed_seed(self) -> None:
        config = RDBPriorConfig(
            min_tables=4,
            max_tables=4,
            min_rows_per_table=32,
            max_rows_per_table=32,
            min_features_per_table=2,
            max_features_per_table=2,
            seed=1,
        )
        first = RelationalPriorGenerator(config).sample_database()
        second = RelationalPriorGenerator(config).sample_database()

        self.assertEqual(first.table_specs, second.table_specs)
        self.assertEqual(first.foreign_keys, second.foreign_keys)
        for table_name in first.tables:
            np.testing.assert_array_equal(first.tables[table_name], second.tables[table_name])
            np.testing.assert_array_equal(
                first.feature_missing_masks[table_name],
                second.feature_missing_masks[table_name],
            )
        for key in first.foreign_key_null_masks:
            np.testing.assert_array_equal(
                first.foreign_key_null_masks[key], second.foreign_key_null_masks[key]
            )

    def test_non_null_foreign_keys_satisfy_referential_integrity(self) -> None:
        database = RelationalPriorGenerator(
            RDBPriorConfig(
                min_tables=5, max_tables=5, min_rows_per_table=40, max_rows_per_table=40, seed=2
            )
        ).sample_database()

        for fk in database.foreign_keys:
            child_values = database.column_values(fk.child_table, fk.child_column)
            parent_values = set(
                database.column_values(fk.parent_table, fk.parent_column).astype(np.int64).tolist()
            )
            for value in child_values[np.isfinite(child_values)]:
                self.assertIn(int(value), parent_values)

    def test_one_to_one_edges_have_max_fanout_one(self) -> None:
        config = RDBPriorConfig(
            min_tables=6,
            max_tables=6,
            min_rows_per_table=40,
            max_rows_per_table=40,
            optional_foreign_key_probability=0.0,
            capacity_limited_probability=0.0,
            one_to_one_probability=1.0,
            temporal_foreign_key_probability=0.0,
            multi_parent_probability=0.0,
            enable_snapshot_tables=True,
            schema_archetype=SchemaArchetypeConfig(forced_archetype="temporal-history"),
            seed=3,
        )
        database = RelationalPriorGenerator(config).sample_database()
        one_to_one_edges = [
            fk
            for fk in database.foreign_keys
            if fk.cardinality == "one_to_one" or fk.mechanism.capacity_mode == "one_to_one"
        ]
        self.assertGreater(len(one_to_one_edges), 0)

        for fk in one_to_one_edges:
            values = database.column_values(fk.child_table, fk.child_column)
            parent_rows = values[np.isfinite(values)].astype(np.int64)
            counts = np.bincount(
                parent_rows, minlength=database.table_specs[fk.parent_table].row_count
            )
            self.assertLessEqual(int(counts.max(initial=0)), 1)

    def test_capacity_limited_edges_do_not_exceed_capacity(self) -> None:
        from rfm.rdb.statistics import generate_forced_regime_database

        database = generate_forced_regime_database("capacity_limited", seed=4)
        capacity_edges = [fk for fk in database.foreign_keys if fk.capacity is not None]
        self.assertGreater(len(capacity_edges), 0)

        for fk in capacity_edges:
            values = database.column_values(fk.child_table, fk.child_column)
            parent_rows = values[np.isfinite(values)].astype(np.int64)
            counts = np.bincount(
                parent_rows, minlength=database.table_specs[fk.parent_table].row_count
            )
            self.assertLessEqual(int(counts.max(initial=0)), fk.capacity)

    def test_optional_edges_have_structural_nulls(self) -> None:
        from rfm.rdb.config import MechanismHyperpriorConfig

        config = RDBPriorConfig(
            min_tables=4,
            max_tables=4,
            min_rows_per_table=80,
            max_rows_per_table=80,
            optional_foreign_key_probability=1.0,
            mechanism_hyperprior=MechanismHyperpriorConfig(forced_existence="optional"),
            capacity_limited_probability=0.0,
            one_to_one_probability=0.0,
            temporal_foreign_key_probability=0.0,
            seed=5,
        )
        database = RelationalPriorGenerator(config).sample_database()

        optional_fks = [
            fk for fk in database.foreign_keys if fk.existence in ("optional", "sparse")
        ]
        self.assertGreater(len(optional_fks), 0)
        for fk in optional_fks:
            observed_rate = float(np.mean(database.foreign_key_null_masks[fk.key]))
            self.assertGreater(observed_rate, 0.02)
            self.assertLess(observed_rate, 0.95)

    def test_edge_metadata_records_semantic_cardinality_and_existence(self) -> None:
        database = RelationalPriorGenerator(RDBPriorConfig(seed=6)).sample_database()
        edge_metadata = database.metadata.get("edge_metadata")
        self.assertIsInstance(edge_metadata, dict)
        self.assertGreater(len(database.foreign_keys), 0)
        for fk in database.foreign_keys:
            self.assertIsNotNone(fk.semantic)
            self.assertIsNotNone(fk.existence)
            self.assertEqual(fk.intent, fk.semantic)
            self.assertEqual(fk.existence, fk.mechanism.existence)
            assert isinstance(edge_metadata, dict)
            self.assertEqual(edge_metadata[fk.key]["semantic"], fk.semantic)
            self.assertEqual(edge_metadata[fk.key]["cardinality"], fk.cardinality)
            self.assertEqual(edge_metadata[fk.key]["existence"], fk.existence)

    def test_nullable_edges_remain_optional_under_forced_mandatory_existence(self) -> None:
        from rfm.rdb.config import MechanismHyperpriorConfig

        config = RDBPriorConfig(
            min_tables=4,
            max_tables=4,
            min_rows_per_table=40,
            max_rows_per_table=40,
            optional_foreign_key_probability=1.0,
            mechanism_hyperprior=MechanismHyperpriorConfig(forced_existence="mandatory"),
            seed=61,
        )
        database = RelationalPriorGenerator(config).sample_database()
        nullable_edges = [fk for fk in database.foreign_keys if fk.nullable]

        self.assertGreater(len(nullable_edges), 0)
        self.assertTrue(all(fk.existence == "optional" for fk in nullable_edges))

    def test_temporal_edges_only_reference_past_parent_rows(self) -> None:
        config = RDBPriorConfig(
            min_tables=5,
            max_tables=5,
            min_rows_per_table=48,
            max_rows_per_table=48,
            optional_foreign_key_probability=0.0,
            capacity_limited_probability=0.0,
            one_to_one_probability=0.0,
            temporal_foreign_key_probability=1.0,
            seed=6,
        )
        database = RelationalPriorGenerator(config).sample_database()

        for fk in database.foreign_keys:
            if not fk.temporal:
                continue
            child_times = database.column_values(fk.child_table, "timestamp")
            parent_times = database.column_values(fk.parent_table, "timestamp")
            values = database.column_values(fk.child_table, fk.child_column)
            for child_row, parent_value in enumerate(values):
                if np.isfinite(parent_value):
                    self.assertLessEqual(
                        float(parent_times[int(parent_value)]), float(child_times[child_row]) + 1e-6
                    )

    def test_multi_parent_router_creates_joint_dependency(self) -> None:
        config = RDBPriorConfig(
            min_tables=4,
            max_tables=4,
            min_rows_per_table=96,
            max_rows_per_table=96,
            optional_foreign_key_probability=0.0,
            capacity_limited_probability=0.0,
            one_to_one_probability=0.0,
            temporal_foreign_key_probability=0.0,
            multi_parent_probability=1.0,
            seed=7,
        )
        database = RelationalPriorGenerator(config).sample_database()
        groups: dict[str, list[str]] = {}
        for fk in database.foreign_keys:
            if fk.multi_parent_group is not None:
                groups.setdefault(fk.multi_parent_group, []).append(fk.key)
        multi_group = next((keys for keys in groups.values() if len(keys) >= 2), None)
        if multi_group is None:
            self.skipTest("no multi-parent group sampled")
        first_table, first_col = multi_group[0].split(".")
        second_table, second_col = multi_group[1].split(".")
        self.assertEqual(first_table, second_table)
        first = database.column_values(first_table, first_col)
        second = database.column_values(second_table, second_col)
        mask = np.isfinite(first) & np.isfinite(second)
        observed = _mutual_information(first[mask].astype(np.int64), second[mask].astype(np.int64))
        perm = np.random.default_rng(7).permutation(int(mask.sum()))
        shuffled = _mutual_information(
            first[mask].astype(np.int64), second[mask].astype(np.int64)[perm]
        )
        self.assertGreater(observed, shuffled + 0.01)

    def test_feature_values_are_finite_except_explicit_missingness(self) -> None:
        database = RelationalPriorGenerator(
            RDBPriorConfig(
                min_tables=5, max_tables=5, min_rows_per_table=36, max_rows_per_table=36, seed=8
            )
        ).sample_database()

        for table_name, spec in database.table_specs.items():
            missing_mask = database.feature_missing_masks[table_name]
            for feature_idx, column in enumerate(spec.feature_columns):
                values = database.column_values(table_name, column.name)
                self.assertFalse(np.isinf(values).any())
                self.assertTrue(np.array_equal(np.isnan(values), missing_mask[:, feature_idx]))

    def test_sample_task_returns_valid_split(self) -> None:
        generator = RelationalPriorGenerator(
            RDBPriorConfig(
                min_tables=4, max_tables=4, min_rows_per_table=32, max_rows_per_table=32, seed=9
            )
        )
        database = generator.sample_database()
        task = generator.sample_task(database)
        row_count = database.table_specs[task.target_table].row_count

        self.assertGreater(len(task.train_indices), 0)
        self.assertGreater(len(task.test_indices), 0)
        self.assertTrue(
            set(task.train_indices.tolist()).isdisjoint(set(task.test_indices.tolist()))
        )
        self.assertTrue(np.all(task.train_indices >= 0))
        self.assertTrue(np.all(task.test_indices < row_count))

    def test_temporal_task_split_requires_target_timestamp(self) -> None:
        role_grammar = RoleGrammarConfig(
            timestamp_probability_by_role={
                "entity": 0.0,
                "activity/event": 0.0,
                "dimension/lookup": 0.0,
                "bridge": 0.0,
                "snapshot/state": 0.0,
            }
        )
        generator = RelationalPriorGenerator(
            RDBPriorConfig(
                min_tables=4,
                max_tables=4,
                min_rows_per_table=32,
                max_rows_per_table=32,
                role_grammar=role_grammar,
                temporal_task_probability=1.0,
                ood_task_probability=0.0,
                explicit_relational_target_probability=0.0,
                seed=10,
            )
        )
        database = generator.sample_database()
        task = generator.sample_task(database)

        self.assertIsNone(database.table_specs[task.target_table].timestamp_column)
        self.assertNotEqual(task.split_kind, "temporal")

    def test_explicit_relational_target_records_causal_spec(self) -> None:
        for seed in range(18, 28):
            generator = RelationalPriorGenerator(
                RDBPriorConfig(
                    min_tables=5,
                    max_tables=5,
                    min_rows_per_table=48,
                    max_rows_per_table=48,
                    explicit_relational_target_probability=1.0,
                    cross_table_target_probability=1.0,
                    seed=seed,
                )
            )
            database = generator.sample_database()
            task = generator.sample_task(database)
            if task.target_spec is not None and task.target_spec.target_family != "local_only":
                self.assertIsNotNone(task.target_values)
                self.assertTrue(task.has_cross_table_path)
                self.assertIn(
                    task.target_spec.target_family,
                    (
                        "parent_feature",
                        "parent_child_interaction",
                        "multi_parent",
                        "topology_driven",
                    ),
                )
                return

        self.fail("could not sample explicit cross-table target")

    def test_explicit_target_difficulty_filter_records_probe_metrics(self) -> None:
        for seed in range(60, 80):
            config = RDBPriorConfig(
                min_tables=5,
                max_tables=5,
                min_rows_per_table=48,
                max_rows_per_table=48,
                explicit_relational_target_probability=1.0,
                seed=seed,
            )
            generator = RelationalPriorGenerator(config)
            database = generator.sample_database()
            task = generator.sample_task(database)
            if task.target_spec is None or task.target_spec.difficulty_metrics is None:
                continue

            metrics = task.target_spec.difficulty_metrics
            if task.task_type == "classification":
                self.assertIsNotNone(metrics.class_entropy)
                self.assertIsNotNone(metrics.min_class_fraction)
                self.assertGreaterEqual(
                    metrics.class_entropy, config.explicit_target_min_class_entropy
                )
                self.assertGreaterEqual(
                    metrics.min_class_fraction, config.explicit_target_min_class_fraction
                )
                self.assertLessEqual(
                    metrics.accepted_score, config.explicit_target_max_classification_probe_accuracy
                )
            else:
                self.assertGreaterEqual(
                    metrics.accepted_score, config.explicit_target_min_regression_probe_r2
                )
                self.assertLessEqual(
                    metrics.accepted_score, config.explicit_target_max_regression_probe_r2
                )
            if metrics.relational_gain is not None:
                self.assertGreaterEqual(
                    metrics.relational_gain, config.explicit_target_min_relational_probe_gain
                )
            return

        self.fail("could not sample explicit target with difficulty metrics")

    def test_classification_task_split_has_matching_classes(self) -> None:
        generator = RelationalPriorGenerator(
            RDBPriorConfig(
                min_tables=5,
                max_tables=5,
                min_rows_per_table=48,
                max_rows_per_table=48,
                feature_types=("categorical", "ordinal", "binary", "quantized"),
                task="classification",
                temporal_task_probability=1.0,
                ood_task_probability=1.0,
                seed=19,
            )
        )
        database = generator.sample_database()
        task = generator.sample_task(database)
        if task.target_values is None:
            values = database.column_values(task.target_table, task.target_column).astype(np.int64)
        else:
            values = task.target_values.astype(np.int64)

        train_classes = set(np.unique(values[task.train_indices]).tolist())
        test_classes = set(np.unique(values[task.test_indices]).tolist())

        self.assertEqual(train_classes, test_classes)
        self.assertGreaterEqual(len(train_classes), 2)

    def test_sample_pretrain_spec_has_neighbors_and_paths(self) -> None:
        spec = RelationalPriorGenerator(
            RDBPriorConfig(
                min_tables=5, max_tables=5, min_rows_per_table=32, max_rows_per_table=32, seed=10
            )
        ).sample_pretrain_spec()
        self.assertIsNotNone(spec.focal_table)
        self.assertIsInstance(spec.neighbor_tables, tuple)
        self.assertIsInstance(spec.join_paths, tuple)
        self.assertIn("attachment_counts", spec.topology_summary)

    def test_structural_null_matches_nan_foreign_keys(self) -> None:
        database = RelationalPriorGenerator(
            RDBPriorConfig(
                min_tables=5, max_tables=6, min_rows_per_table=48, max_rows_per_table=48, seed=11
            )
        ).sample_database()
        for fk in database.foreign_keys:
            values = database.column_values(fk.child_table, fk.child_column)
            null_mask = database.foreign_key_null_masks[fk.key]
            self.assertTrue(np.array_equal(np.isnan(values), null_mask))

    def test_snapshot_constraint_keeps_latest_valid_parents(self) -> None:
        gate = CandidateConstraintGate()
        parent_times = np.array([0.1, 0.4, 0.7, 0.9], dtype=np.float64)
        fanout = np.zeros(4, dtype=np.int64)
        fk = _minimal_fk()
        candidates = gate.candidates(
            fk=fk,
            child_row=0,
            parent_count=4,
            child_time=1.0,
            parent_timestamps=parent_times,
            fanout=fanout,
            snapshot_latest_only=True,
        )
        self.assertEqual(candidates.tolist(), [3])

    def test_joint_tuple_rejects_empty_candidate_lists(self) -> None:
        rng = np.random.default_rng(12)
        config = RDBPriorConfig(seed=12)
        sampler = JointParentTupleSampler(rng, config, RelationFieldScorer(rng).scores)
        fk = _minimal_fk()
        with self.assertRaises(ValueError):
            sampler._sample_joint_product(
                child_row=0,
                fks=(fk, fk),
                candidate_lists=(np.array([], dtype=np.int64), np.array([1], dtype=np.int64)),
                child_latent=np.zeros(8),
                row_latents={"parent": np.zeros((4, 8))},
                child_time=0.5,
                parent_timestamps={"parent": np.linspace(0.0, 1.0, 4)},
                fanout={fk.key: np.zeros(4, dtype=np.int64)},
                dynamic_popularity={"parent": np.ones(4)},
            )

    def test_snapshot_table_foreign_keys_use_latest_valid_parent_time(self) -> None:
        for seed in range(40, 55):
            database = RelationalPriorGenerator(
                RDBPriorConfig(
                    min_tables=5,
                    max_tables=6,
                    min_rows_per_table=32,
                    max_rows_per_table=32,
                    temporal_foreign_key_probability=1.0,
                    enable_snapshot_tables=True,
                    schema_archetype=SchemaArchetypeConfig(forced_archetype="temporal-history"),
                    role_grammar=RoleGrammarConfig(
                        timestamp_probability_by_role={
                            "entity": 1.0,
                            "activity/event": 1.0,
                            "dimension/lookup": 1.0,
                            "bridge": 1.0,
                            "snapshot/state": 1.0,
                        }
                    ),
                    seed=seed,
                )
            ).sample_database()
            snapshot_tables = [
                name for name, spec in database.table_specs.items() if spec.role == "snapshot/state"
            ]
            if not snapshot_tables:
                continue
            for table_name in snapshot_tables:
                for fk in database.foreign_keys:
                    if fk.child_table != table_name or not fk.temporal:
                        continue
                    child_times = database.column_values(table_name, "timestamp")
                    parent_times = database.column_values(fk.parent_table, "timestamp")
                    values = database.column_values(fk.child_table, fk.child_column)
                    for child_row, parent_value in enumerate(values):
                        if not np.isfinite(parent_value):
                            continue
                        child_time = float(child_times[child_row])
                        parent_row = int(parent_value)
                        parent_time = float(parent_times[parent_row])
                        self.assertLessEqual(parent_time, child_time + 1e-6)
                        valid_times = parent_times[parent_times <= child_time + 1e-12]
                        self.assertAlmostEqual(parent_time, float(np.max(valid_times)), places=6)
                    return
        self.skipTest("no snapshot table with temporal foreign keys in seed sweep")

    def test_small_topology_context_dim_is_supported(self) -> None:
        database = RelationalPriorGenerator(
            RDBPriorConfig(
                min_tables=4,
                max_tables=4,
                min_rows_per_table=32,
                max_rows_per_table=32,
                topology_context_dim=3,
                seed=56,
            )
        ).sample_database()

        self.assertEqual(len(database.table_specs), 4)


def _minimal_fk() -> ForeignKeySpec:
    profile = MechanismProfile(
        existence="mandatory",
        attachment="uniform",
        coordination="independent",
        field_weights=(0.0,) * 8,
        temperature=1.0,
        capacity_mode="unbounded",
        existence_latent_weight=(0.0,) * 8,
    )
    return ForeignKeySpec(
        child_table="child",
        child_column="parent_id",
        parent_table="parent",
        parent_column="parent_id",
        cardinality="many_to_one",
        nullable=False,
        capacity=None,
        temporal=True,
        mechanism=profile,
        intent="snapshot_refs_entity_or_activity",
    )


def _mutual_information(x: NDArray[np.int64], y: NDArray[np.int64]) -> float:
    if len(x) == 0:
        return 0.0
    x_values, x_inverse = np.unique(x, return_inverse=True)
    y_values, y_inverse = np.unique(y, return_inverse=True)
    joint = np.zeros((len(x_values), len(y_values)), dtype=np.float64)
    np.add.at(joint, (x_inverse, y_inverse), 1.0)
    joint = joint / float(len(x))
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = joint > 0.0
    return float(np.sum(joint[mask] * np.log(joint[mask] / expected[mask])))


if __name__ == "__main__":
    unittest.main()
