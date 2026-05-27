from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from relbench.base import Database

from plurel import RFMSyntheticDataset
from rfm.rdb import RDBPriorConfig, RelationalSCMConfig


def _small_rfm_config(seed: int) -> RDBPriorConfig:
    return RDBPriorConfig(
        min_tables=4,
        max_tables=4,
        min_rows_per_table=24,
        max_rows_per_table=24,
        min_features_per_table=2,
        max_features_per_table=2,
        feature_types=("continuous", "binary"),
        relational_scm=RelationalSCMConfig(
            feature_missing_probability=0.0,
            max_feature_missing_rate=0.0,
        ),
        seed=seed,
    )


def test_rfm_synthetic_dataset_returns_relbench_database(tmp_path: Path) -> None:
    cache_dir = tmp_path / "rel-synthetic-rfm-101"
    dataset = RFMSyntheticDataset(seed=101, config=_small_rfm_config(101), cache_dir=cache_dir)

    db = dataset.get_db()

    assert isinstance(db, Database)
    assert db.table_dict
    metadata = json.loads((cache_dir / "rfm_metadata.json").read_text())
    assert metadata["seed"] == 101
    assert metadata["schema_archetype"] is not None
    assert metadata["table_roles"]
    assert metadata["mechanism_counts"]
    assert metadata["mandatory_fk_stats"]


def test_rfm_relbench_tables_preserve_keys_timestamps_and_features(tmp_path: Path) -> None:
    dataset = RFMSyntheticDataset(
        seed=102,
        config=_small_rfm_config(102),
        cache_dir=tmp_path / "rel-synthetic-rfm-102",
    )
    db = dataset.get_db()
    trainable_feature_columns = []

    for table_name, table in db.table_dict.items():
        df = table.df
        assert table.pkey_col is not None
        assert df[table.pkey_col].dropna().astype(int).tolist() == list(range(len(df)))

        if table.time_col is not None:
            assert pd.api.types.is_datetime64_any_dtype(df[table.time_col])

        feature_columns = [column for column in df.columns if column.startswith("feature_")]
        assert feature_columns
        for column in feature_columns:
            dtype = df[column].dtype
            assert pd.api.types.is_float_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
            if pd.api.types.is_float_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
                trainable_feature_columns.append((table_name, column))

        for fk_col, parent_table_name in table.fkey_col_to_pkey_table.items():
            parent_table = db.table_dict[parent_table_name]
            assert parent_table.pkey_col is not None
            parent_keys = set(parent_table.df[parent_table.pkey_col].dropna().astype(int).tolist())
            fk_values = df[fk_col].dropna().astype(int).tolist()
            assert all(value in parent_keys for value in fk_values)

    assert trainable_feature_columns
