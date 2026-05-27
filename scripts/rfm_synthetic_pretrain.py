from __future__ import annotations

import argparse
import os
from pathlib import Path

from rt.main import main as rt_main
from rt.tasks import generate_rel_synthetic_tasks_from_db_names


def _rfm_db_names(seed_offset: int, count: int) -> list[str]:
    return [f"rel-synthetic-rfm-{seed_offset + idx}" for idx in range(count)]


def main(
    seed_offset: int,
    num_train_dbs: int,
    num_test_dbs: int,
    cache_root: Path,
    pre_root: Path,
    max_steps: int,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    ctx_len: int,
    max_bfs_width: int,
    save_ckpt_dir: Path | None,
    skip_reg_tasks: bool = False,
    skip_clf_tasks: bool = False,
) -> None:
    cache_root = cache_root.expanduser()
    pre_root = pre_root.expanduser()
    os.environ["PLUREL_PRE_ROOT"] = str(pre_root)

    test_db_names = _rfm_db_names(seed_offset=seed_offset, count=num_test_dbs)
    train_db_names = _rfm_db_names(
        seed_offset=seed_offset + num_test_dbs,
        count=num_train_dbs,
    )
    rel_synthetic_tasks = generate_rel_synthetic_tasks_from_db_names(
        train_db_names=train_db_names,
        test_db_names=test_db_names,
        cache_root=cache_root,
        skip_reg_tasks=skip_reg_tasks,
        skip_clf_tasks=skip_clf_tasks,
        backend="rfm",
    )
    train_tasks = (
        rel_synthetic_tasks["train_autocomplete_clf_tasks"]
        + rel_synthetic_tasks["train_autocomplete_reg_tasks"]
    )
    eval_tasks = (
        rel_synthetic_tasks["test_autocomplete_clf_tasks"]
        + rel_synthetic_tasks["test_autocomplete_reg_tasks"]
    )
    if not train_tasks:
        raise ValueError("RFM synthetic pretraining found no train tasks")
    if not eval_tasks:
        raise ValueError("RFM synthetic pretraining found no eval tasks")

    rt_main(
        project="rt-rfm-synthetic",
        eval_splits=["val", "test"],
        eval_freq=max(max_steps - 1, 1),
        eval_pow2=False,
        max_eval_steps=1,
        load_ckpt_path=None,
        save_ckpt_dir=str(save_ckpt_dir.expanduser()) if save_ckpt_dir is not None else None,
        compile_=False,
        seed=0,
        train_tasks=train_tasks,
        eval_tasks=eval_tasks,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        ctx_len=ctx_len,
        max_bfs_width=max_bfs_width,
        lr=5e-4,
        lr_schedule=True,
        wd=0.1,
        max_grad_norm=1.0,
        max_steps=max_steps,
        embedding_model="all-MiniLM-L12-v2",
        d_text=384,
        num_blocks=12,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        cohort="rfm",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RT pretraining on RFM synthetic DBs.")
    parser.add_argument("--seed_offset", type=int, default=14000)
    parser.add_argument("--num_train_dbs", type=int, required=True)
    parser.add_argument("--num_test_dbs", type=int, required=True)
    parser.add_argument("--cache_root", type=Path, default=Path("~/.cache/relbench"))
    parser.add_argument("--pre_root", type=Path, default=Path("~/scratch/pre"))
    parser.add_argument("--max_steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--ctx_len", type=int, default=1024)
    parser.add_argument("--max_bfs_width", type=int, default=128)
    parser.add_argument(
        "--save_ckpt_dir",
        type=Path,
        default=None,
        help="Optional checkpoint directory.",
    )
    parser.add_argument("--skip_reg_tasks", action="store_true")
    parser.add_argument("--skip_clf_tasks", action="store_true")
    args = parser.parse_args()

    main(
        seed_offset=args.seed_offset,
        num_train_dbs=args.num_train_dbs,
        num_test_dbs=args.num_test_dbs,
        cache_root=args.cache_root,
        pre_root=args.pre_root,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        ctx_len=args.ctx_len,
        max_bfs_width=args.max_bfs_width,
        save_ckpt_dir=args.save_ckpt_dir,
        skip_reg_tasks=args.skip_reg_tasks,
        skip_clf_tasks=args.skip_clf_tasks,
    )
