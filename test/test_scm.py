import warnings
from types import MethodType

import numpy as np
import pandas as pd
import pytest
from torch_frame import stype

from plurel.config import DAGParams, SCMParams
from plurel.dag import DAG_REGISTRY
from plurel.scm import SCM, SOURCE_GEN_REGISTRY
from plurel.utils import TableType


def _make_scm(seed: int = 0, child_table_names: list[str] | None = None) -> SCM:
    return SCM(
        table_name="t",
        child_table_names=child_table_names or [],
        feature_columns={
            "num_col": {"_stype": stype.numerical, "categories": None},
            "cat_col": {"_stype": stype.categorical, "categories": [0, 1, 2]},
        },
        pkey_col="id",
        fkey_col_to_pkey_table={},
        foreign_scm_info={},
        scm_params=SCMParams(),
        dag_params=DAGParams(),
        seed=seed,
    )


@pytest.mark.parametrize("seed", list(range(5)))
def test_scm(seed):
    scms = []
    table_ids = list(range(len(DAG_REGISTRY)))
    dag_classes = list(DAG_REGISTRY.keys())
    for table_id, dag_class in zip(table_ids, dag_classes):
        child_table_ids = table_ids[table_id + 1 :]
        parent_table_ids = table_ids[:table_id]

        scm = SCM(
            table_name=f"table_{table_id}",
            child_table_names=[f"table_{c_id}" for c_id in child_table_ids],
            feature_columns={
                "feature_0": {"_stype": stype.numerical, "categories": None},
                "feature_1": {
                    "_stype": stype.categorical,
                    "categories": ["active", "inactive"],
                },
            },
            pkey_col="row_idx",
            fkey_col_to_pkey_table={},
            foreign_scm_info={},
            scm_params=SCMParams(),
            dag_params=DAGParams(),
            seed=seed * 100 + table_id,
        )
        df = scm.generate_df(
            num_rows=10,
            table_type=(TableType.Entity if np.random.rand() < 0.5 else TableType.Activity),
        )
        assert len(df) == 10
        scms.append(scm)


def test_multiple_fks_to_same_parent(monkeypatch):
    parent_scm = _make_scm(seed=1, child_table_names=["child"])
    parent_scm.generate_df(num_rows=8, table_type=TableType.Entity)

    call_count = {"value": 0}

    def fake_sample_bipartite_assignments(
        size_a: int, size_b: int, hierarchy_a: list[int], hierarchy_b: list[int]
    ) -> np.ndarray:
        value = call_count["value"]
        call_count["value"] += 1
        return np.full(size_b, value, dtype=np.int64)

    monkeypatch.setattr(
        "plurel.topology_prior.sample_bipartite_assignments", fake_sample_bipartite_assignments
    )

    child_scm = SCM(
        table_name="child",
        child_table_names=[],
        feature_columns={},
        pkey_col="id",
        fkey_col_to_pkey_table={"fka": "parent", "fkb": "parent"},
        foreign_scm_info={"parent": parent_scm},
        scm_params=SCMParams(),
        dag_params=DAGParams(),
        seed=2,
    )
    df = child_scm.generate_df(num_rows=6, table_type=TableType.Entity)

    assert df["fka"].tolist() == [0] * 6
    assert df["fkb"].tolist() == [1] * 6
    assert not df["fka"].equals(df["fkb"])


def test_null_fk_propagation_smoke():
    parent_scm = _make_scm(seed=3, child_table_names=["child"])
    parent_scm.generate_df(num_rows=8, table_type=TableType.Entity)

    child_scm = SCM(
        table_name="child",
        child_table_names=[],
        feature_columns={},
        pkey_col="id",
        fkey_col_to_pkey_table={"fka": "parent"},
        foreign_scm_info={"parent": parent_scm},
        scm_params=SCMParams(),
        dag_params=DAGParams(),
        seed=4,
    )

    def fake_initialize_bi_fk_pk_graph_map(self):
        self.foreign_row_idxs_map = {("fka", "parent"): np.array([0, 1, 2, 3], dtype=np.int64)}
        self.foreign_null_mask_map = {
            ("fka", "parent"): np.array([False, True, False, False], dtype=bool)
        }

    child_scm.initialize_bi_fk_pk_graph_map = MethodType(
        fake_initialize_bi_fk_pk_graph_map, child_scm
    )

    df = child_scm.generate_df(num_rows=4, table_type=TableType.Activity)

    assert str(df["fka"].dtype) == "Int64"
    assert df.loc[0, "fka"] == 0
    assert df.loc[1, "fka"] is pd.NA
    assert pd.isna(df.loc[1, "fka"])
    assert df.loc[2, "fka"] == 2
    assert df.loc[3, "fka"] == 3


@pytest.mark.parametrize("gen_type", list(SOURCE_GEN_REGISTRY.keys()))
def test_source_gen_registry_numerical_finite(gen_type):
    np.random.seed(0)
    factory = SOURCE_GEN_REGISTRY[gen_type]
    gen = factory.make_numerical(
        scm_params=SCMParams(), num_rows=100, table_type=TableType.Activity
    )
    values = [gen.get_value(row_idx=i) for i in range(100)]
    assert all(np.isfinite(v) for v in values), (
        f"source gen '{gen_type}' produced non-finite values"
    )


@pytest.mark.parametrize("seed", range(10))
def test_scm_generate_df_no_inf_nan(seed):
    df = _make_scm(seed=seed).generate_df(num_rows=50, table_type=TableType.Entity)
    float_cols = df.select_dtypes(float)
    assert not float_cols.isin([float("inf"), float("-inf")]).any().any()
    assert not float_cols.isna().any().any()


@pytest.mark.parametrize("seed", range(10))
def test_scm_generate_df_no_overflow_warning(seed):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _make_scm(seed=seed).generate_df(num_rows=50, table_type=TableType.Entity)
