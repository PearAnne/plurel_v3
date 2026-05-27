from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import trace_rt_sampling_budget as trace_budget


def _make_trace_batch() -> dict[str, object]:
    return {
        "true_batch_size": 3,
        "trace_cells_used": torch.tensor([120, 1024, 900, 0], dtype=torch.int64),
        "trace_ctx_full": torch.tensor([False, True, False, False], dtype=torch.bool),
        "trace_bfs_raw_cells_seen": torch.tensor([160, 1200, 1000, 0], dtype=torch.int64),
        "trace_bfs_cell_budget_hit": torch.tensor([False, True, False, False], dtype=torch.bool),
        "trace_bfs_width_events": torch.tensor([0, 2, 0, 0], dtype=torch.int64),
        "trace_bfs_width_candidates": torch.tensor([10, 300, 80, 0], dtype=torch.int64),
        "trace_bfs_width_selected": torch.tensor([10, 128, 80, 0], dtype=torch.int64),
        "trace_bfs_width_dropped": torch.tensor([0, 172, 0, 0], dtype=torch.int64),
        "trace_bfs_width_max_candidates": torch.tensor([10, 300, 80, 0], dtype=torch.int64),
        "trace_f2p_slots_max": torch.tensor([1, 5, 3, 0], dtype=torch.int64),
    }


def test_trace_accumulator_summarizes_runtime_budget() -> None:
    accumulator = trace_budget.TraceAccumulator(ctx_len=1024)
    accumulator.update(batch=_make_trace_batch(), true_batch_size=3)
    accumulator.update(
        batch={
            "true_batch_size": 2,
            "trace_cells_used": torch.tensor([64, 64], dtype=torch.int64),
            "trace_ctx_full": torch.tensor([False, False], dtype=torch.bool),
            "trace_bfs_raw_cells_seen": torch.tensor([80, 70], dtype=torch.int64),
            "trace_bfs_cell_budget_hit": torch.tensor([False, False], dtype=torch.bool),
            "trace_bfs_width_events": torch.tensor([1, 0], dtype=torch.int64),
            "trace_bfs_width_candidates": torch.tensor([140, 30], dtype=torch.int64),
            "trace_bfs_width_selected": torch.tensor([128, 30], dtype=torch.int64),
            "trace_bfs_width_dropped": torch.tensor([12, 0], dtype=torch.int64),
            "trace_bfs_width_max_candidates": torch.tensor([140, 30], dtype=torch.int64),
            "trace_f2p_slots_max": torch.tensor([5, 2], dtype=torch.int64),
        },
        true_batch_size=2,
    )

    summary = accumulator.summary(
        cohort="G0_hsbm",
        max_bfs_width=128,
        batch_size=4,
        requested_batches=2,
    )

    assert summary.observed_batches == 2
    assert summary.observed_sequences == 5
    assert summary.ctx_full_rate == 0.2
    assert summary.bfs_cell_budget_hit_rate == 0.2
    assert summary.bfs_width_truncated_sequence_rate == 0.4
    assert summary.bfs_width_dropped_total == 184
    assert summary.f2p_slots_max_observed == 5
    assert summary.f2p_at_limit_sequence_rate == 0.4
    assert summary.cells_used.max == 1024.0


def test_run_matrix_expands_setting_grid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "manifest.json").write_text(
        """
        {
          "cohorts": [
            {"cohort": "G0_hsbm", "seeds": [30000, 30001, 30002, 30003], "num_dbs": 4},
            {"cohort": "G7_realistic_mix", "seeds": [30000, 30001, 30002, 30003], "num_dbs": 4}
          ]
        }
        """,
        encoding="utf-8",
    )

    task_map = {
        "G0_hsbm": [("db", "table", "target", [])],
        "G7_realistic_mix": [("db2", "table2", "target2", [])],
    }

    def fake_collect_runtime_budget_summary(**kwargs):
        return trace_budget.RuntimeBudgetSummary(
            cohort=kwargs["cohort"],
            max_bfs_width=kwargs["max_bfs_width"],
            ctx_len=kwargs["ctx_len"],
            batch_size=kwargs["batch_size"],
            requested_batches=kwargs["max_batches"],
            observed_batches=1,
            observed_sequences=1,
            cells_used=trace_budget.NumericSummary(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            padding_ratio_mean=0.0,
            ctx_full_rate=0.0,
            bfs_cell_budget_hit_rate=0.0,
            bfs_width_truncated_sequence_rate=0.0,
            bfs_width_events_total=0,
            bfs_width_events_per_sequence=0.0,
            bfs_width_candidates_total=0,
            bfs_width_selected_total=0,
            bfs_width_dropped_total=0,
            bfs_width_dropped_ratio=0.0,
            bfs_width_max_candidates=trace_budget.NumericSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            f2p_slots_max_observed=0,
            f2p_at_limit_sequence_rate=0.0,
        )

    def fake_build_train_tasks_from_pre_root(train_db_names, pre_root, per_db_task_limit):
        cohort = "G0_hsbm" if "G0_hsbm" in train_db_names[0] else "G7_realistic_mix"
        return task_map[cohort]

    monkeypatch.setattr(
        trace_budget,
        "build_train_tasks_from_pre_root",
        fake_build_train_tasks_from_pre_root,
    )
    monkeypatch.setattr(
        trace_budget, "collect_runtime_budget_summary", fake_collect_runtime_budget_summary
    )

    args = SimpleNamespace(
        manifest=tmp_path / "manifest.json",
        cohorts=["G0_hsbm", "G7_realistic_mix"],
        num_train_dbs=2,
        num_test_dbs=1,
        max_bfs_widths=[128, 512],
        ctx_lens=[1024, 2048],
        batch_size=16,
        max_batches=1,
        seed=0,
        pre_root=tmp_path / "pre",
        per_db_task_limit=None,
        embedding_model="all-MiniLM-L12-v2",
        d_text=384,
        batch_trace_path=None,
    )

    payload = trace_budget.run_matrix(args)

    assert len(payload["results"]) == 8
    assert {row["max_bfs_width"] for row in payload["results"]} == {128, 512}
    assert {row["ctx_len"] for row in payload["results"]} == {1024, 2048}
