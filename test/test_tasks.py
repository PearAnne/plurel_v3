from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from rt import tasks


@dataclass
class _FakeTable:
    df: pd.DataFrame
    fkey_col_to_pkey_table: dict[str, str]


@dataclass
class _FakeDb:
    table_dict: dict[str, _FakeTable]


class _FakeSyntheticDataset:
    calls: list[tuple[int, str]] = []

    def __init__(self, seed: int, config: Any) -> None:
        self.seed = seed
        self.config = config
        self.calls.append((seed, str(config.cache_dir)))

    def get_db(self) -> _FakeDb:
        return _FakeDb(
            table_dict={
                "table_a": _FakeTable(
                    df=pd.DataFrame(
                        {
                            "feature_bool": pd.Series([True, False], dtype=bool),
                            "feature_float": pd.Series([1.0, 2.0], dtype=float),
                            "ignored": pd.Series([1, 2], dtype=int),
                        }
                    ),
                    fkey_col_to_pkey_table={},
                )
            }
        )


class _FakeRFMSyntheticDataset:
    calls: list[tuple[int, str]] = []

    def __init__(self, seed: int, cache_dir: Any) -> None:
        self.seed = seed
        self.cache_dir = cache_dir
        self.calls.append((seed, str(cache_dir)))

    def get_db(self) -> _FakeDb:
        return _FakeDb(
            table_dict={
                "table_rfm": _FakeTable(
                    df=pd.DataFrame(
                        {
                            "feature_bool": pd.Series([True, False], dtype="boolean"),
                            "feature_float": pd.Series([1.0, 2.0], dtype=float),
                        }
                    ),
                    fkey_col_to_pkey_table={},
                )
            }
        )


def test_generate_rel_synthetic_tasks_from_db_names_uses_explicit_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _FakeSyntheticDataset.calls = []
    monkeypatch.setattr(tasks, "SyntheticDataset", _FakeSyntheticDataset)

    bundle = tasks.generate_rel_synthetic_tasks_from_db_names(
        train_db_names=["rel-synthetic-G7_realistic_mix-16001"],
        test_db_names=["rel-synthetic-G7_realistic_mix-16000"],
        cache_root=tmp_path,
    )

    assert bundle["train_autocomplete_clf_tasks"] == [
        ("rel-synthetic-G7_realistic_mix-16001", "table_a", "feature_bool", [])
    ]
    assert bundle["train_autocomplete_reg_tasks"] == [
        ("rel-synthetic-G7_realistic_mix-16001", "table_a", "feature_float", [])
    ]
    assert bundle["test_autocomplete_clf_tasks"] == [
        ("rel-synthetic-G7_realistic_mix-16000", "table_a", "feature_bool", [])
    ]
    assert [call[0] for call in _FakeSyntheticDataset.calls] == [16000, 16001]
    assert _FakeSyntheticDataset.calls[0][1].endswith("rel-synthetic-G7_realistic_mix-16000")
    assert _FakeSyntheticDataset.calls[1][1].endswith("rel-synthetic-G7_realistic_mix-16001")


def test_generate_rel_synthetic_tasks_from_db_names_can_skip_task_types(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(tasks, "SyntheticDataset", _FakeSyntheticDataset)

    bundle = tasks.generate_rel_synthetic_tasks_from_db_names(
        train_db_names=["rel-synthetic-G0_hsbm-16001"],
        test_db_names=["rel-synthetic-G0_hsbm-16000"],
        cache_root=tmp_path,
        skip_clf_tasks=True,
    )

    assert bundle["train_autocomplete_clf_tasks"] == []
    assert bundle["test_autocomplete_clf_tasks"] == []
    assert bundle["train_autocomplete_reg_tasks"]
    assert bundle["test_autocomplete_reg_tasks"]


def test_generate_rel_synthetic_tasks_from_db_names_supports_rfm_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _FakeRFMSyntheticDataset.calls = []
    monkeypatch.setattr(tasks, "RFMSyntheticDataset", _FakeRFMSyntheticDataset)

    bundle = tasks.generate_rel_synthetic_tasks_from_db_names(
        train_db_names=["rel-synthetic-rfm-16001"],
        test_db_names=["rel-synthetic-rfm-16000"],
        cache_root=tmp_path,
        backend="rfm",
    )

    assert bundle["train_autocomplete_clf_tasks"] == [
        ("rel-synthetic-rfm-16001", "table_rfm", "feature_bool", [])
    ]
    assert bundle["train_autocomplete_reg_tasks"] == [
        ("rel-synthetic-rfm-16001", "table_rfm", "feature_float", [])
    ]
    assert bundle["test_autocomplete_clf_tasks"] == [
        ("rel-synthetic-rfm-16000", "table_rfm", "feature_bool", [])
    ]
    assert [call[0] for call in _FakeRFMSyntheticDataset.calls] == [16000, 16001]
    assert _FakeRFMSyntheticDataset.calls[0][1].endswith("rel-synthetic-rfm-16000")
    assert _FakeRFMSyntheticDataset.calls[1][1].endswith("rel-synthetic-rfm-16001")


def test_get_tasks_info_skips_degenerate_targets() -> None:
    db = _FakeDb(
        table_dict={
            "table_a": _FakeTable(
                df=pd.DataFrame(
                    {
                        "feature_bool_constant": pd.Series([True, True, pd.NA], dtype="boolean"),
                        "feature_bool_valid": pd.Series([True, False, pd.NA], dtype="boolean"),
                        "feature_float_constant": pd.Series([1.0, 1.0, 1.0], dtype=float),
                        "feature_float_valid": pd.Series([1.0, 2.0, float("nan")], dtype=float),
                    }
                ),
                fkey_col_to_pkey_table={},
            )
        }
    )

    tasks_info = tasks.get_tasks_info(db=db, db_name="rel-synthetic-rfm-1", table_name="table_a")

    assert tasks_info["clf"] == [("rel-synthetic-rfm-1", "table_a", "feature_bool_valid", [])]
    assert tasks_info["reg"] == [("rel-synthetic-rfm-1", "table_a", "feature_float_valid", [])]
