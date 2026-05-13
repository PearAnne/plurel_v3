from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from plurel.topology_adapters import (
    _infer_time_column,
    load_ctu_database,
    load_dbinfer_database,
    parse_ctu_fkey_column,
)
from plurel.topology_measure import (
    TopologyDatabase,
    TopologyTable,
    fk_values_to_parent_row_indices,
    measure_topology_database,
)


def test_parse_ctu_fkey_column_resolves_parent_table_and_pkey():
    table_names = {"district", "upravna_enota", "nesreca"}
    assert parse_ctu_fkey_column("FK_district_district_id", table_names) == (
        "district",
        "district_id",
    )
    assert parse_ctu_fkey_column("FK_upravna_enota_upravna_enota", table_names) == (
        "upravna_enota",
        "upravna_enota",
    )


def test_fk_values_to_parent_row_indices_maps_parent_keys_to_row_positions():
    parent_df = pd.DataFrame({"client_id": [10, 20, 30]})
    fk = pd.Series([20, 10, None, 99], dtype="Int64")
    parent_idx, null_mask = fk_values_to_parent_row_indices(fk, parent_df, "client_id")
    assert parent_idx.tolist() == [1, 0, 0, 0]
    assert null_mask.tolist() == [False, False, True, True]


def test_fk_values_to_parent_row_indices_handles_duplicate_parent_keys():
    parent_df = pd.DataFrame({"k": [1, 1, 2], "x": [0, 1, 2]})
    fk = pd.Series([1, 2])
    parent_idx, null_mask = fk_values_to_parent_row_indices(fk, parent_df, "k")
    assert null_mask.tolist() == [False, False]
    assert parent_idx.tolist() == [0, 2]


def test_infer_time_column_skips_lap_duration_string_time_column():
    df = pd.DataFrame(
        {
            "time": ["1:49.088", "1:50.012", "1:51.100"],
            "milliseconds": [109088, 110012, 111100],
        }
    )
    assert _infer_time_column(df) == "milliseconds"


def test_infer_time_column_parses_grant_style_string_dates():
    df = pd.DataFrame({"award_effective_date": ["07/01/1986", "01/15/1987", "03/20/1988"]})
    assert _infer_time_column(df) == "award_effective_date"


def test_measure_topology_database_with_no_foreign_keys_counts_table_rows():
    database = TopologyDatabase(
        db_name="ctu-empty-fk",
        tables={"t": TopologyTable(df=pd.DataFrame({"a": [1, 2, 3]}))},
        foreign_keys=[],
    )
    stats = measure_topology_database(database)
    assert stats.num_edges == 0
    assert stats.total_child_rows == 3
    assert stats.total_non_null_edges == 0


@pytest.mark.skipif(
    not Path("/local/lzd/plurel_runtime/relbench/ctu/financial/db").exists(),
    reason="CTU mirror not available locally",
)
def test_load_ctu_database_financial_discovers_fk_edges():
    database = load_ctu_database(
        ctu_root=Path("/local/lzd/plurel_runtime/relbench/ctu"),
        db_name="financial",
    )
    assert database.db_name == "ctu-financial"
    assert database.foreign_keys
    stats = measure_topology_database(database)
    assert stats.num_edges == len(database.foreign_keys)
    assert stats.total_non_null_edges > 0


@pytest.mark.skipif(
    not Path("/local/lzd/plurel_runtime/relbench/dbinfer-seznam/db").exists(),
    reason="4DBInfer mirror not available locally",
)
def test_load_dbinfer_database_seznam_loads_declared_edges():
    database = load_dbinfer_database(
        data_root=Path("/local/lzd/plurel_runtime/relbench"),
        dataset_name="seznam",
    )
    assert database.db_name == "dbinfer-seznam"
    assert len(database.foreign_keys) == 3
    stats = measure_topology_database(database)
    assert stats.num_edges == 3
