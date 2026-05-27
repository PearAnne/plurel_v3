from __future__ import annotations

import argparse
import os
import subprocess
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import torch
from tqdm import tqdm

from plurel import RFMSyntheticDataset, make_rt_compatible_rfm_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def generate_rfm_synthetic_db(
    seed: int,
    cache_root: Path,
    pre_root: Path,
    preprocess: bool = False,
    embed: bool = True,
) -> None:
    torch.set_num_threads(1)
    db_name = f"rel-synthetic-rfm-{seed}"
    print(f"Creating dataset: {db_name}")

    cache_root = cache_root.expanduser()
    pre_root = pre_root.expanduser()
    dataset = RFMSyntheticDataset(
        seed=seed,
        config=make_rt_compatible_rfm_config(seed=seed),
        cache_dir=cache_root / db_name,
    )
    dataset.get_db()

    if not preprocess:
        return

    rustler_dir = REPO_ROOT / "rustler"
    subprocess.run(
        [
            "pixi",
            "run",
            "cargo",
            "run",
            "--release",
            "--",
            "pre",
            db_name,
            "--data-root",
            str(cache_root),
            "--pre-root",
            str(pre_root),
        ],
        cwd=rustler_dir,
        check=True,
    )

    if not embed:
        return

    env = {**os.environ, "PLUREL_PRE_ROOT": str(pre_root)}
    subprocess.run(
        ["pixi", "run", "python", "-m", "rt.embed", db_name],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def main(
    seed_offset: int,
    num_dbs: int,
    num_proc: int,
    cache_root: Path,
    pre_root: Path,
    preprocess: bool = False,
    embed: bool = True,
) -> None:
    seeds = [idx + seed_offset for idx in range(num_dbs)]
    worker = partial(
        generate_rfm_synthetic_db,
        cache_root=cache_root,
        pre_root=pre_root,
        preprocess=preprocess,
        embed=embed,
    )

    with Pool(processes=num_proc) as pool:
        list(tqdm(pool.imap_unordered(worker, seeds), total=len(seeds)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RFM RDB synthetic datasets.")
    parser.add_argument(
        "--seed_offset",
        type=int,
        required=True,
        help="Seed offset. DBs will be named rel-synthetic-rfm-<seed>.",
    )
    parser.add_argument("--num_dbs", type=int, required=True, help="Number of databases.")
    parser.add_argument(
        "--num_proc",
        type=int,
        default=cpu_count(),
        help="Number of parallel processes.",
    )
    parser.add_argument(
        "--cache_root",
        type=Path,
        default=Path("~/.cache/relbench"),
        help="RelBench cache root.",
    )
    parser.add_argument(
        "--pre_root",
        type=Path,
        default=Path("~/scratch/pre"),
        help="Rustler pre output root.",
    )
    parser.add_argument(
        "--preprocess",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run rustler pre and text embedding after DB generation.",
    )
    parser.add_argument(
        "--embed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run rt.embed after rustler pre when --preprocess is enabled.",
    )
    args = parser.parse_args()

    main(
        seed_offset=args.seed_offset,
        num_dbs=args.num_dbs,
        num_proc=args.num_proc,
        cache_root=args.cache_root,
        pre_root=args.pre_root,
        preprocess=args.preprocess,
        embed=args.embed,
    )
