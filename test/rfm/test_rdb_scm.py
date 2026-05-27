from __future__ import annotations

import unittest

import numpy as np

from rfm.rdb import RDBPriorConfig, RelationalPriorGenerator, RelationalSCMGenerator
from rfm.rdb.types import ColumnSpec, TableSpec
from rfm.scm import ExogenousContext, SCMGenerator


class RelationalSCMTest(unittest.TestCase):
    def test_feature_values_finite_except_missing(self) -> None:
        database = RelationalPriorGenerator(
            RDBPriorConfig(
                min_tables=4, max_tables=4, min_rows_per_table=32, max_rows_per_table=32, seed=31
            )
        ).sample_database()
        for table_name, spec in database.table_specs.items():
            missing_mask = database.feature_missing_masks[table_name]
            for feature_idx, column in enumerate(spec.feature_columns):
                values = database.column_values(table_name, column.name)
                self.assertFalse(np.isinf(values).any())
                self.assertTrue(np.array_equal(np.isnan(values), missing_mask[:, feature_idx]))

    def test_exogenous_ablation_changes_distribution(self) -> None:
        rng = np.random.default_rng(32)
        config = RDBPriorConfig(seed=32)
        generator = RelationalSCMGenerator(rng, config)
        table = TableSpec(
            name="entity_0",
            role="entity",
            row_count=48,
            columns=(
                ColumnSpec(name="entity_0_id", kind="primary_key", value_type="integer"),
                ColumnSpec(name="timestamp", kind="timestamp", value_type="timestamp"),
                ColumnSpec(name="entity_0_f0", kind="feature", value_type="continuous"),
            ),
            primary_key="entity_0_id",
            timestamp_column="timestamp",
        )
        num_rows = table.row_count
        row_context = rng.normal(size=(num_rows, config.latent_dim + 1))
        parent_zero = np.zeros((num_rows, config.latent_dim))
        parent_signal = row_context[:, : config.latent_dim] * 3.0 + rng.normal(
            scale=0.1, size=(num_rows, config.latent_dim)
        )
        topology = rng.normal(size=(num_rows, config.topology_context_dim))

        with_parent = generator.sample_table_columns(
            table, row_context, parent_signal, topology
        ).values["entity_0_f0"]
        without_parent = generator.sample_table_columns(
            table, row_context, parent_zero, topology
        ).values["entity_0_f0"]
        self.assertFalse(
            np.allclose(
                with_parent[np.isfinite(with_parent)],
                without_parent[np.isfinite(without_parent)],
                rtol=0.05,
                atol=0.05,
            )
        )

    def test_scm_exogenous_injection(self) -> None:
        rng = np.random.default_rng(33)
        scm = SCMGenerator(rng, RDBPriorConfig().relational_scm.prior)
        num_rows = 40
        exogenous = ExogenousContext(
            row_context=rng.normal(size=(num_rows, 4)),
            parent_context=rng.normal(size=(num_rows, 3)),
            topology_context=rng.normal(size=(num_rows, 2)),
        )
        spec = scm.sample_spec_with_exogenous(num_nodes=3, num_exogenous=9, node_dim=1)
        values = scm.sample_values(
            spec, num_rows=num_rows, train_size=20, temporal=False, ood=False, exogenous=exogenous
        )
        self.assertTrue(np.isfinite(values[:, : spec.num_exogenous, :]).all())

    def test_relational_content_preserves_schema_feature_count(self) -> None:
        rng = np.random.default_rng(34)
        config = RDBPriorConfig(
            min_features_per_table=7,
            max_features_per_table=7,
            seed=34,
        )
        generator = RelationalSCMGenerator(rng, config)
        columns = (
            ColumnSpec(name="entity_0_id", kind="primary_key", value_type="integer"),
            ColumnSpec(name="timestamp", kind="timestamp", value_type="timestamp"),
            ColumnSpec(name="entity_0_f0", kind="feature", value_type="continuous"),
            ColumnSpec(name="entity_0_f1", kind="feature", value_type="categorical"),
            ColumnSpec(name="entity_0_f2", kind="feature", value_type="ordinal"),
            ColumnSpec(name="entity_0_f3", kind="feature", value_type="binary"),
            ColumnSpec(name="entity_0_f4", kind="feature", value_type="count"),
            ColumnSpec(name="entity_0_f5", kind="feature", value_type="quantized"),
            ColumnSpec(
                name="entity_0_f6", kind="feature", value_type="high_cardinality_categorical"
            ),
        )
        table = TableSpec(
            name="entity_0",
            role="entity",
            row_count=40,
            columns=columns,
            primary_key="entity_0_id",
            timestamp_column="timestamp",
        )
        row_context = rng.normal(size=(table.row_count, config.latent_dim + 1))
        parent_context = rng.normal(size=(table.row_count, config.latent_dim))
        topology_context = rng.normal(size=(table.row_count, config.topology_context_dim))

        result = generator.sample_table_columns(
            table, row_context, parent_context, topology_context
        )

        self.assertEqual(set(result.values), {column.name for column in table.feature_columns})
        self.assertEqual(result.missing_mask.shape, (table.row_count, len(table.feature_columns)))


if __name__ == "__main__":
    unittest.main()
