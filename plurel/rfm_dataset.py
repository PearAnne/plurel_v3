from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from relbench.base import Database, Dataset, Table

from rfm.rdb import RDBPriorConfig, RelationalDataset, RelationalPriorGenerator
from rfm.rdb.types import ColumnSpec, TableSpec

RFM_TIME_START = pd.Timestamp("1990-01-01")
RFM_TIME_END = pd.Timestamp("2025-01-01")


def make_rt_compatible_rfm_config(seed: int | None = None) -> RDBPriorConfig:
    return RDBPriorConfig(seed=seed, feature_types=("continuous", "binary"))


class RFMSyntheticDataset(Dataset):
    """RelBench-compatible wrapper around the RFM RDB prior."""

    def __init__(
        self,
        seed: int,
        config: RDBPriorConfig | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.seed = seed
        self.config = replace(config or make_rt_compatible_rfm_config(), seed=seed)
        self.cache_dir_path = Path(cache_dir).expanduser() if cache_dir is not None else None
        self.min_timestamp = RFM_TIME_START
        self.max_timestamp = RFM_TIME_END
        self.val_timestamp = _fractional_timestamp(0.8)
        self.test_timestamp = _fractional_timestamp(0.9)
        super().__init__(
            cache_dir=str(self.cache_dir_path) if self.cache_dir_path is not None else None
        )

    def make_db(self) -> Database:
        relational_db = RelationalPriorGenerator(
            config=self.config, seed=self.seed
        ).sample_database()
        db = rfm_relational_dataset_to_relbench(relational_db)
        self._write_metadata(relational_db)
        return db

    def _write_metadata(self, relational_db: RelationalDataset) -> None:
        if self.cache_dir_path is None:
            return

        metadata = {
            "seed": self.seed,
            "config": _to_jsonable(asdict(self.config)),
            "schema_archetype": relational_db.metadata.get("schema_archetype"),
            "table_roles": relational_db.metadata.get("table_roles"),
            "mechanism_counts": relational_db.metadata.get("mechanism_counts"),
            "mandatory_fk_stats": relational_db.metadata.get("mandatory_fk_stats"),
        }
        self.cache_dir_path.mkdir(parents=True, exist_ok=True)
        metadata_path = self.cache_dir_path / "rfm_metadata.json"
        metadata_path.write_text(
            json.dumps(_to_jsonable(metadata), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def rfm_relational_dataset_to_relbench(relational_db: RelationalDataset) -> Database:
    table_dict = {}
    for table_name, table_spec in relational_db.table_specs.items():
        df = _table_to_dataframe(relational_db, table_spec)
        fkey_col_to_pkey_table = {
            fk.child_column: fk.parent_table
            for fk in relational_db.foreign_keys
            if fk.child_table == table_name
        }
        table_dict[table_name] = Table(
            df=df,
            fkey_col_to_pkey_table=fkey_col_to_pkey_table,
            pkey_col=table_spec.primary_key,
            time_col=table_spec.timestamp_column,
        )
    return Database(table_dict)


def _table_to_dataframe(relational_db: RelationalDataset, table_spec: TableSpec) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    feature_idx = 0
    for column in table_spec.columns:
        values = relational_db.column_values(table_spec.name, column.name)
        if column.kind == "primary_key":
            data[column.name] = _nullable_int_series(values, column.name)
        elif column.kind == "foreign_key":
            data[column.name] = _nullable_int_series(values, column.name)
        elif column.kind == "timestamp":
            data[column.name] = _datetime_series(values, column.name)
        elif column.kind == "feature":
            output_name = f"feature_{feature_idx}"
            data[output_name] = _feature_series(values, column, output_name)
            feature_idx += 1
        else:
            raise ValueError(f"unknown RFM column kind {column.kind!r}")
    return pd.DataFrame(data)


def _nullable_int_series(values: NDArray[np.float32], name: str) -> pd.Series:
    array = np.asarray(values, dtype=np.float64)
    series = pd.Series(pd.NA, index=np.arange(len(array)), dtype="Int64", name=name)
    finite = np.isfinite(array)
    if finite.any():
        series.loc[finite] = np.rint(array[finite]).astype(np.int64)
    return series


def _datetime_series(values: NDArray[np.float32], name: str) -> pd.Series:
    array = np.asarray(values, dtype=np.float64)
    series = pd.Series(pd.NaT, index=np.arange(len(array)), dtype="datetime64[ns]", name=name)
    finite = np.isfinite(array)
    if finite.any():
        clipped = np.clip(array[finite], 0.0, 1.0)
        duration_ns = RFM_TIME_END.value - RFM_TIME_START.value
        timestamps = RFM_TIME_START.value + np.rint(clipped * duration_ns).astype(np.int64)
        series.loc[finite] = pd.to_datetime(timestamps)
    return series


def _feature_series(values: NDArray[np.float32], column: ColumnSpec, name: str) -> pd.Series:
    array = np.asarray(values, dtype=np.float64)
    if column.value_type == "binary":
        series = pd.Series(pd.NA, index=np.arange(len(array)), dtype="boolean", name=name)
        finite = np.isfinite(array)
        if finite.any():
            series.loc[finite] = array[finite] > 0.0
        return series
    if column.value_type == "continuous":
        return pd.Series(array.astype(np.float64, copy=False), name=name)

    series = pd.Series(pd.NA, index=np.arange(len(array)), dtype="Int64", name=name)
    finite = np.isfinite(array)
    if finite.any():
        series.loc[finite] = np.rint(array[finite]).astype(np.int64)
    return series


def _fractional_timestamp(fraction: float) -> pd.Timestamp:
    duration_ns = RFM_TIME_END.value - RFM_TIME_START.value
    return pd.Timestamp(RFM_TIME_START.value + int(round(duration_ns * fraction)))


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
