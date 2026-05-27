from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rt.tasks import forecast_tasks, generate_rel_synthetic_tasks_from_db_names

DEFAULT_EVAL_DBS = {"rel-hm", "rel-avito", "rel-stack", "rel-trial", "rel-f1", "rel-amazon"}
DEFAULT_EVAL_FREQ = 400


@dataclass(frozen=True)
class CohortRunSpec:
    cohort: str
    train_db_names: list[str]
    test_db_names: list[str]
    save_ckpt_dir: Path
    rt_kwargs: dict[str, Any]


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


def build_rt_kwargs(
    cohort: str,
    train_db_names: list[str],
    test_db_names: list[str],
    cache_root: Path,
    save_ckpt_dir: Path,
    max_steps: int,
    max_bfs_width: int,
    seed: int,
    batch_size: int,
    eval_batch_size: int,
    ctx_len: int,
    eval_freq: int = DEFAULT_EVAL_FREQ,
) -> dict[str, Any]:
    if eval_freq <= 0:
        raise ValueError(f"eval_freq must be positive, got {eval_freq}")
    rel_synthetic_tasks = generate_rel_synthetic_tasks_from_db_names(
        train_db_names=train_db_names,
        test_db_names=test_db_names,
        cache_root=cache_root,
    )
    train_tasks = (
        rel_synthetic_tasks["train_autocomplete_clf_tasks"]
        + rel_synthetic_tasks["train_autocomplete_reg_tasks"]
    )
    eval_tasks = [task for task in forecast_tasks if task[0] in DEFAULT_EVAL_DBS]
    eval_tasks += rel_synthetic_tasks["test_autocomplete_clf_tasks"]
    eval_tasks += rel_synthetic_tasks["test_autocomplete_reg_tasks"]
    return {
        "project": "rt",
        "eval_splits": ["val", "test"],
        "eval_freq": eval_freq,
        "eval_pow2": False,
        "max_eval_steps": 80,
        "load_ckpt_path": None,
        "save_ckpt_dir": str(save_ckpt_dir),
        "compile_": True,
        "seed": seed,
        "cohort": cohort,
        "train_tasks": train_tasks,
        "eval_tasks": eval_tasks,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "num_workers": 2,
        "ctx_len": ctx_len,
        "max_bfs_width": max_bfs_width,
        "lr": 5e-4,
        "lr_schedule": True,
        "wd": 0.1,
        "max_grad_norm": 1.0,
        "max_steps": max_steps,
        "embedding_model": "all-MiniLM-L12-v2",
        "d_text": 384,
        "num_blocks": 12,
        "d_model": 256,
        "num_heads": 8,
        "d_ff": 1024,
    }


def build_run_specs(
    manifest_path: Path,
    cohorts: list[str],
    num_train_dbs: int,
    num_test_dbs: int,
    max_steps: int,
    max_bfs_width: int,
    cache_root: Path,
    save_root: Path,
    seed: int,
    batch_size: int,
    eval_batch_size: int,
    ctx_len: int,
    eval_freq: int = DEFAULT_EVAL_FREQ,
) -> list[CohortRunSpec]:
    manifest = load_manifest(manifest_path)
    rows = select_cohort_rows(manifest=manifest, cohorts=cohorts)
    specs = []
    for row in rows:
        db_names = cohort_db_names(row)
        train_db_names, test_db_names = split_train_test_db_names(
            db_names=db_names,
            num_train_dbs=num_train_dbs,
            num_test_dbs=num_test_dbs,
        )
        save_ckpt_dir = (
            save_root.expanduser()
            / row["cohort"]
            / (
                f"train{num_train_dbs}_test{num_test_dbs}_steps{max_steps}_"
                f"bfs{max_bfs_width}_bs{batch_size}_ctx{ctx_len}_seed{seed}"
            )
        )
        rt_kwargs = build_rt_kwargs(
            cohort=row["cohort"],
            train_db_names=train_db_names,
            test_db_names=test_db_names,
            cache_root=cache_root,
            save_ckpt_dir=save_ckpt_dir,
            max_steps=max_steps,
            max_bfs_width=max_bfs_width,
            seed=seed,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            ctx_len=ctx_len,
            eval_freq=eval_freq,
        )
        specs.append(
            CohortRunSpec(
                cohort=row["cohort"],
                train_db_names=train_db_names,
                test_db_names=test_db_names,
                save_ckpt_dir=save_ckpt_dir,
                rt_kwargs=rt_kwargs,
            )
        )
    return specs


def render_dry_run(specs: list[CohortRunSpec]) -> str:
    lines = []
    for spec in specs:
        block = [
            f"cohort={spec.cohort}",
            f"  train_db_names={spec.train_db_names}",
            f"  test_db_names={spec.test_db_names}",
            f"  save_ckpt_dir={spec.save_ckpt_dir}",
            f"  train_tasks={len(spec.rt_kwargs['train_tasks'])}",
            f"  eval_tasks={len(spec.rt_kwargs['eval_tasks'])}",
            f"  max_steps={spec.rt_kwargs['max_steps']}",
            f"  eval_freq={spec.rt_kwargs['eval_freq']}",
            f"  max_bfs_width={spec.rt_kwargs['max_bfs_width']}",
        ]
        if "batch_size" in spec.rt_kwargs:
            block.append(f"  batch_size={spec.rt_kwargs['batch_size']}")
        if "ctx_len" in spec.rt_kwargs:
            block.append(f"  ctx_len={spec.rt_kwargs['ctx_len']}")
        lines.extend(block)
    return "\n".join(lines)


def run_specs(specs: list[CohortRunSpec], pre_root: Path, dry_run: bool) -> None:
    os.environ["PLUREL_PRE_ROOT"] = str(pre_root.expanduser().resolve())
    if dry_run:
        print(render_dry_run(specs))
        return
    _ensure_python_bin_on_path()
    os.environ.setdefault("WANDB_MODE", "disabled")
    from rt.main import main as rt_main

    for spec in specs:
        rt_main(**spec.rt_kwargs)


def _ensure_python_bin_on_path() -> None:
    python_bin = Path(sys.executable).resolve().parent
    python_env = python_bin.parent
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if str(python_bin) not in path_entries:
        os.environ["PATH"] = os.pathsep.join([str(python_bin), *path_entries])
    os.environ.setdefault("VIRTUAL_ENV", str(python_env))
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix is not None and Path(conda_prefix).resolve() != python_env:
        os.environ.pop("CONDA_PREFIX", None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cohort-aware edge-prior RT pretraining.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cohorts", nargs="+", required=True)
    parser.add_argument("--num_train_dbs", type=int, default=8)
    parser.add_argument("--num_test_dbs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=401)
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=DEFAULT_EVAL_FREQ,
        help="Evaluate every N training steps. Keep this small enough to see learning curves.",
    )
    parser.add_argument("--max_bfs_width", type=int, default=128)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Training batch size (lower if CUDA OOM).",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=128,
        help="Eval batch size (lower if CUDA OOM during validation).",
    )
    parser.add_argument(
        "--ctx_len",
        type=int,
        default=1024,
        help="Context length (lower reduces VRAM).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cache_root", type=Path, default=Path("/local/lzd/plurel_runtime/relbench")
    )
    parser.add_argument("--pre_root", type=Path, default=Path("/local/lzd/plurel_runtime/pre"))
    parser.add_argument("--save_root", type=Path, default=Path("/local/lzd/plurel_runtime/rt_runs"))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    specs = build_run_specs(
        manifest_path=args.manifest,
        cohorts=args.cohorts,
        num_train_dbs=args.num_train_dbs,
        num_test_dbs=args.num_test_dbs,
        max_steps=args.max_steps,
        eval_freq=args.eval_freq,
        max_bfs_width=args.max_bfs_width,
        cache_root=args.cache_root,
        save_root=args.save_root,
        seed=args.seed,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        ctx_len=args.ctx_len,
    )
    run_specs(specs=specs, pre_root=args.pre_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
