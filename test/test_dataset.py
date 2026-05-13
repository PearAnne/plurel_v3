import numpy as np
import pytest

from plurel.config import Choices, Config, DatabaseParams, SCMParams
from plurel.dataset import COLUMN_TRANSFORM_REGISTRY, SyntheticDataset

_TRANSFORM_INPUTS = {
    "standard_normal": np.random.default_rng(0).standard_normal(1000),
    "all_zeros": np.zeros(100),
    "large_positive": np.full(100, 1e6),
    "large_negative": np.full(100, -1e6),
    "mixed_sign": np.linspace(-1e4, 1e4, 1000),
}


@pytest.mark.parametrize("seed", list(range(100)))
def test_dataset(seed):
    config = Config(
        database_params=DatabaseParams(
            num_tables_choices=Choices(kind="range", value=[1, 5]),
            num_rows_entity_table_choices=Choices(kind="range", value=[40, 80]),
            num_rows_activity_table_choices=Choices(kind="range", value=[100, 200]),
        )
    )
    dataset = SyntheticDataset(seed=seed, config=config)
    db = dataset.make_db()
    assert db is not None


@pytest.mark.parametrize("seed", list(range(20)))
def test_dataset_with_sql_file(seed, schema_sql):
    config = Config(
        database_params=DatabaseParams(
            num_rows_entity_table_choices=Choices(kind="range", value=[40, 80]),
            num_rows_activity_table_choices=Choices(kind="range", value=[100, 200]),
        ),
        schema_file=schema_sql,
    )
    dataset = SyntheticDataset(seed=seed, config=config)
    db = dataset.make_db()
    assert db is not None


@pytest.mark.parametrize("transform_name", list(COLUMN_TRANSFORM_REGISTRY.keys()))
@pytest.mark.parametrize("input_name", list(_TRANSFORM_INPUTS.keys()))
def test_column_transform_finite(transform_name, input_name):
    x = _TRANSFORM_INPUTS[input_name]
    result = COLUMN_TRANSFORM_REGISTRY[transform_name](x)
    assert np.all(np.isfinite(result)), (
        f"transform '{transform_name}' on '{input_name}' produced non-finite values"
    )


@pytest.mark.parametrize("prior_kind", ["hsbm", "erdos_renyi", "chung_lu", "dcsbm", "tpa"])
def test_dataset_make_db_with_topology_prior(prior_kind):
    config = Config(
        database_params=DatabaseParams(
            num_tables_choices=Choices(kind="range", value=[3, 3]),
            num_rows_entity_table_choices=Choices(kind="range", value=[20, 30]),
            num_rows_activity_table_choices=Choices(kind="range", value=[40, 60]),
            column_nan_perc_choices=Choices(kind="range", value=[0.0, 0.0]),
        ),
        scm_params=SCMParams(
            topology_prior_choices=Choices(kind="set", value=[prior_kind]),
            edge_prior_null_rate_choices=Choices(kind="range", value=[0.0, 0.0]),
            propagate_batch_size=128,
        ),
    )
    dataset = SyntheticDataset(seed=0, config=config)

    db = dataset.make_db()

    assert db is not None
    for table in db.table_dict.values():
        assert len(table.df) > 0
        for fkey_col in table.fkey_col_to_pkey_table:
            assert str(table.df[fkey_col].dtype) in {"int64", "Int64"}
