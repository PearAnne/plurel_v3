from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from scripts.run_edge_prior_pretrain import _ensure_python_bin_on_path

_ensure_python_bin_on_path()

from rt.data import RelationalDataset

LOGGER = logging.getLogger(__name__)
MAX_F2P_NBRS = 5
TRACE_REQUIRED_KEYS = (
    "trace_cells_used",
    "trace_ctx_full",
    "trace_bfs_raw_cells_seen",
    "trace_bfs_cell_budget_hit",
    "trace_bfs_width_events",
    "trace_bfs_width_candidates",
    "trace_bfs_width_selected",
    "trace_bfs_width_dropped",
    "trace_bfs_width_max_candidates",
    "trace_f2p_slots_max",
)


@dataclass(frozen=True)
class NumericSummary:
    mean: float
    p50: float
    p90: float
    p95: float
    p99: float
    max: float


@dataclass(frozen=True)
class RuntimeBudgetSummary:
    cohort: str
    max_bfs_width: int
    ctx_len: int
    batch_size: int
    requested_batches: int
    observed_batches: int
    observed_sequences: int
    cells_used: NumericSummary
    padding_ratio_mean: float
    ctx_full_rate: float
    bfs_cell_budget_hit_rate: float
    bfs_width_truncated_sequence_rate: float
    bfs_width_events_total: int
    bfs_width_events_per_sequence: float
    bfs_width_candidates_total: int
    bfs_width_selected_total: int
    bfs_width_dropped_total: int
    bfs_width_dropped_ratio: float
    bfs_width_max_candidates: NumericSummary
    f2p_slots_max_observed: int
    f2p_at_limit_sequence_rate: float


class TraceAccumulator:
    def __init__(self, ctx_len: int) -> None:
        self.ctx_len = ctx_len
        self.observed_batches = 0
        self.observed_sequences = 0
        self.cells_used: list[int] = []
        self.bfs_width_max_candidates: list[int] = []
        self.ctx_full_count = 0
        self.bfs_cell_budget_hit_count = 0
        self.bfs_width_truncated_sequence_count = 0
        self.bfs_width_events_total = 0
        self.bfs_width_candidates_total = 0
        self.bfs_width_selected_total = 0
        self.bfs_width_dropped_total = 0
        self.f2p_slots_max_observed = 0
        self.f2p_at_limit_sequence_count = 0

    def update(self, batch: dict[str, Any], true_batch_size: int) -> dict[str, Any]:
        missing = [key for key in TRACE_REQUIRED_KEYS if key not in batch]
        if missing:
            raise RuntimeError(
                "Sampler trace keys are missing. Rebuild rustler after adding trace support: "
                f"{missing}"
            )

        cells_used = _int_values(batch["trace_cells_used"], true_batch_size)
        ctx_full = _bool_values(batch["trace_ctx_full"], true_batch_size)
        bfs_budget_hit = _bool_values(batch["trace_bfs_cell_budget_hit"], true_batch_size)
        width_events = _int_values(batch["trace_bfs_width_events"], true_batch_size)
        width_candidates = _int_values(batch["trace_bfs_width_candidates"], true_batch_size)
        width_selected = _int_values(batch["trace_bfs_width_selected"], true_batch_size)
        width_dropped = _int_values(batch["trace_bfs_width_dropped"], true_batch_size)
        width_max_candidates = _int_values(batch["trace_bfs_width_max_candidates"], true_batch_size)
        f2p_slots_max = _int_values(batch["trace_f2p_slots_max"], true_batch_size)

        self.observed_batches += 1
        self.observed_sequences += true_batch_size
        self.cells_used.extend(cells_used)
        self.bfs_width_max_candidates.extend(width_max_candidates)
        self.ctx_full_count += sum(ctx_full)
        self.bfs_cell_budget_hit_count += sum(bfs_budget_hit)
        self.bfs_width_truncated_sequence_count += sum(value > 0 for value in width_events)
        self.bfs_width_events_total += sum(width_events)
        self.bfs_width_candidates_total += sum(width_candidates)
        self.bfs_width_selected_total += sum(width_selected)
        self.bfs_width_dropped_total += sum(width_dropped)
        self.f2p_slots_max_observed = max(
            self.f2p_slots_max_observed, max(f2p_slots_max, default=0)
        )
        self.f2p_at_limit_sequence_count += sum(value >= MAX_F2P_NBRS for value in f2p_slots_max)

        return {
            "true_batch_size": true_batch_size,
            "cells_used_mean": _mean(cells_used),
            "ctx_full_count": sum(ctx_full),
            "bfs_cell_budget_hit_count": sum(bfs_budget_hit),
            "bfs_width_events": sum(width_events),
            "bfs_width_dropped": sum(width_dropped),
            "bfs_width_max_candidates": max(width_max_candidates, default=0),
        }

    def summary(
        self,
        cohort: str,
        max_bfs_width: int,
        batch_size: int,
        requested_batches: int,
    ) -> RuntimeBudgetSummary:
        if self.observed_sequences == 0:
            raise RuntimeError(f"No runtime trace was collected for cohort={cohort}")
        return RuntimeBudgetSummary(
            cohort=cohort,
            max_bfs_width=max_bfs_width,
            ctx_len=self.ctx_len,
            batch_size=batch_size,
            requested_batches=requested_batches,
            observed_batches=self.observed_batches,
            observed_sequences=self.observed_sequences,
            cells_used=_numeric_summary(self.cells_used),
            padding_ratio_mean=_mean(
                [(self.ctx_len - value) / self.ctx_len for value in self.cells_used]
            ),
            ctx_full_rate=self.ctx_full_count / self.observed_sequences,
            bfs_cell_budget_hit_rate=self.bfs_cell_budget_hit_count / self.observed_sequences,
            bfs_width_truncated_sequence_rate=(
                self.bfs_width_truncated_sequence_count / self.observed_sequences
            ),
            bfs_width_events_total=self.bfs_width_events_total,
            bfs_width_events_per_sequence=self.bfs_width_events_total / self.observed_sequences,
            bfs_width_candidates_total=self.bfs_width_candidates_total,
            bfs_width_selected_total=self.bfs_width_selected_total,
            bfs_width_dropped_total=self.bfs_width_dropped_total,
            bfs_width_dropped_ratio=_safe_ratio(
                self.bfs_width_dropped_total,
                self.bfs_width_candidates_total,
            ),
            bfs_width_max_candidates=_numeric_summary(self.bfs_width_max_candidates),
            f2p_slots_max_observed=self.f2p_slots_max_observed,
            f2p_at_limit_sequence_rate=self.f2p_at_limit_sequence_count / self.observed_sequences,
        )


def collect_runtime_budget_summary(
    *,
    cohort: str,
    train_tasks: list[tuple[str, str, str, list[str]]],
    batch_size: int,
    ctx_len: int,
    max_bfs_width: int,
    seed: int,
    embedding_model: str,
    d_text: int,
    max_batches: int,
    batch_trace_handle: Any | None = None,
) -> RuntimeBudgetSummary:
    dataset = RelationalDataset(
        tasks=[
            (db_name, table_name, target_column, "train", columns_to_drop)
            for db_name, table_name, target_column, columns_to_drop in train_tasks
        ],
        batch_size=batch_size,
        rank=0,
        world_size=1,
        ctx_len=ctx_len,
        max_bfs_width=max_bfs_width,
        embedding_model=embedding_model,
        d_text=d_text,
        seed=seed,
        trace_sampling=True,
    )
    batches_to_collect = min(max_batches, len(dataset))
    accumulator = TraceAccumulator(ctx_len=ctx_len)

    for batch_idx in range(batches_to_collect):
        batch = dataset[batch_idx]
        true_batch_size = int(batch["true_batch_size"])
        if true_batch_size <= 0:
            continue
        batch_summary = accumulator.update(batch=batch, true_batch_size=true_batch_size)
        if batch_trace_handle is not None:
            payload = {
                "cohort": cohort,
                "max_bfs_width": max_bfs_width,
                "ctx_len": ctx_len,
                "batch_idx": batch_idx,
                **batch_summary,
            }
            batch_trace_handle.write(json.dumps(payload, sort_keys=True) + "\n")

    return accumulator.summary(
        cohort=cohort,
        max_bfs_width=max_bfs_width,
        batch_size=batch_size,
        requested_batches=max_batches,
    )


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["PLUREL_PRE_ROOT"] = str(args.pre_root.expanduser().resolve())
    results: list[RuntimeBudgetSummary] = []
    batch_trace_handle = None
    if args.batch_trace_path is not None:
        args.batch_trace_path.expanduser().parent.mkdir(parents=True, exist_ok=True)
        batch_trace_handle = args.batch_trace_path.expanduser().open("w", encoding="utf-8")

    try:
        manifest = load_manifest(args.manifest)
        cohort_rows = select_cohort_rows(manifest=manifest, cohorts=args.cohorts)
        train_tasks_by_cohort = {
            row["cohort"]: build_train_tasks_from_pre_root(
                train_db_names=split_train_test_db_names(
                    db_names=cohort_db_names(row),
                    num_train_dbs=args.num_train_dbs,
                    num_test_dbs=args.num_test_dbs,
                )[0],
                pre_root=args.pre_root,
                per_db_task_limit=args.per_db_task_limit,
            )
            for row in cohort_rows
        }
        for ctx_len in args.ctx_lens:
            for max_bfs_width in args.max_bfs_widths:
                for cohort in args.cohorts:
                    LOGGER.info(
                        "collecting cohort=%s max_bfs_width=%s ctx_len=%s",
                        cohort,
                        max_bfs_width,
                        ctx_len,
                    )
                    summary = collect_runtime_budget_summary(
                        cohort=cohort,
                        train_tasks=train_tasks_by_cohort[cohort],
                        batch_size=args.batch_size,
                        ctx_len=ctx_len,
                        max_bfs_width=max_bfs_width,
                        seed=args.seed,
                        embedding_model=args.embedding_model,
                        d_text=args.d_text,
                        max_batches=args.max_batches,
                        batch_trace_handle=batch_trace_handle,
                    )
                    results.append(summary)
    finally:
        if batch_trace_handle is not None:
            batch_trace_handle.close()

    return {
        "schema_version": 1,
        "settings": {
            "manifest": str(args.manifest),
            "cohorts": args.cohorts,
            "num_train_dbs": args.num_train_dbs,
            "num_test_dbs": args.num_test_dbs,
            "max_bfs_widths": args.max_bfs_widths,
            "ctx_lens": args.ctx_lens,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "seed": args.seed,
            "per_db_task_limit": args.per_db_task_limit,
            "embedding_model": args.embedding_model,
            "d_text": args.d_text,
            "pre_root": str(args.pre_root),
        },
        "results": [asdict(result) for result in results],
    }


def write_summary(payload: dict[str, Any], output_path: Path) -> None:
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("wrote runtime budget summary to %s", output_path)


def _int_values(tensor: torch.Tensor, true_batch_size: int) -> list[int]:
    return [int(value) for value in tensor[:true_batch_size].cpu().tolist()]


def _bool_values(tensor: torch.Tensor, true_batch_size: int) -> list[bool]:
    return [bool(value) for value in tensor[:true_batch_size].cpu().tolist()]


def _mean(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _numeric_summary(values: list[int]) -> NumericSummary:
    if not values:
        return NumericSummary(mean=0.0, p50=0.0, p90=0.0, p95=0.0, p99=0.0, max=0.0)
    sorted_values = sorted(values)
    return NumericSummary(
        mean=_mean(sorted_values),
        p50=float(_nearest_rank(sorted_values, 0.50)),
        p90=float(_nearest_rank(sorted_values, 0.90)),
        p95=float(_nearest_rank(sorted_values, 0.95)),
        p99=float(_nearest_rank(sorted_values, 0.99)),
        max=float(sorted_values[-1]),
    )


def _nearest_rank(sorted_values: list[int], quantile: float) -> int:
    if not sorted_values:
        return 0
    index = max(0, min(len(sorted_values) - 1, int(quantile * len(sorted_values) + 0.999999) - 1))
    return sorted_values[index]


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.expanduser().read_text(encoding="utf-8"))


def cohort_db_names(cohort_row: dict[str, Any]) -> list[str]:
    cohort_name = cohort_row["cohort"]
    return [f"rel-synthetic-{cohort_name}-{seed}" for seed in cohort_row["seeds"]]


def select_cohort_rows(manifest: dict[str, Any], cohorts: list[str]) -> list[dict[str, Any]]:
    rows_by_name = {row["cohort"]: row for row in manifest.get("cohorts", [])}
    missing = [cohort for cohort in cohorts if cohort not in rows_by_name]
    if missing:
        raise ValueError(f"Unknown cohorts in manifest: {missing}")
    return [rows_by_name[cohort] for cohort in cohorts]


def split_train_test_db_names(
    db_names: list[str],
    num_train_dbs: int,
    num_test_dbs: int,
) -> tuple[list[str], list[str]]:
    required = num_train_dbs + num_test_dbs
    if len(db_names) < required:
        raise ValueError(f"Need {required} DBs, got {len(db_names)}")
    test_db_names = db_names[:num_test_dbs]
    train_db_names = db_names[num_test_dbs:required]
    return train_db_names, test_db_names


def build_train_tasks_from_pre_root(
    train_db_names: list[str],
    pre_root: Path,
    per_db_task_limit: int | None,
) -> list[tuple[str, str, str, list[str]]]:
    tasks: list[tuple[str, str, str, list[str]]] = []
    for db_name in train_db_names:
        column_index_path = pre_root.expanduser() / db_name / "column_index.json"
        column_index = _load_json(column_index_path)
        db_tasks: list[tuple[str, str, str, list[str]]] = []
        for key in sorted(column_index):
            column_name, table_name = _split_column_key(key)
            if not column_name.startswith("feature_"):
                continue
            db_tasks.append((db_name, table_name, column_name, []))
        if per_db_task_limit is not None:
            db_tasks = db_tasks[:per_db_task_limit]
        tasks.extend(db_tasks)
    return tasks


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def _split_column_key(key: str) -> tuple[str, str]:
    if " of " not in key:
        raise ValueError(f"Unexpected column index key: {key}")
    column_name, table_name = key.split(" of ", maxsplit=1)
    return column_name, table_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect runtime RT sampler budget traces for paired cohort settings."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cohorts", nargs="+", required=True)
    parser.add_argument("--num_train_dbs", type=int, default=32)
    parser.add_argument("--num_test_dbs", type=int, default=2)
    parser.add_argument("--max_bfs_widths", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--ctx_lens", nargs="+", type=int, default=[1024, 2048])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_batches", type=int, default=64)
    parser.add_argument(
        "--per_db_task_limit",
        type=int,
        default=None,
        help="Optional cap on feature tasks per synthetic DB for faster diagnostics.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pre_root",
        type=Path,
        default=Path("/local/lzd/plurel_runtime/edge_prior_main_20260517/pre"),
    )
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L12-v2")
    parser.add_argument("--d_text", type=int, default=384)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--batch_trace_path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    payload = run_matrix(args)
    write_summary(payload=payload, output_path=args.output_path)


if __name__ == "__main__":
    main()
