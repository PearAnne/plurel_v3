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
