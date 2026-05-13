from __future__ import annotations

import argparse
import subprocess
from dataclasses import replace
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import torch
from ablation_manifest import build_edge_prior_cohorts, write_manifest
from measure_synthetic_topology import main as measure_synthetic_topology
from tqdm import tqdm

from plurel.dataset import SyntheticDataset
from plurel.utils import set_random_seed

VALID_STAGES = {"gen", "pre", "embed", "pretrain", "eval"}


def parse_stages(raw_stages: str) -> list[str]:
    stages = [stage.strip() for stage in raw_stages.split(",") if stage.strip()]
    unknown = sorted(set(stages) - VALID_STAGES)
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}. Valid stages: {sorted(VALID_STAGES)}")
    return stages


def db_name_for(cohort: str, seed: int) -> str:
    return f"rel-synthetic-{cohort}-{seed}"


def generate_one(cohort_name: str, seed: int, cache_root: Path) -> str:
    torch.set_num_threads(1)
    set_random_seed(0)
    cohorts = build_edge_prior_cohorts()
    spec = cohorts[cohort_name]
    db_name = db_name_for(cohort=cohort_name, seed=seed)
    config = replace(spec.config, cache_dir=str(cache_root.expanduser() / db_name))
    dataset = SyntheticDataset(seed=seed, config=config)
    dataset.get_db()
    return db_name


def run_preprocess_stage(db_names: list[str], stage: str) -> None:
    rustler_dir = Path("rustler").resolve()
    for db_name in db_names:
        if stage == "pre":
            subprocess.run(
                ["pixi", "run", "cargo", "run", "--release", "--", "pre", db_name],
                cwd=rustler_dir,
                check=True,
            )
        elif stage == "embed":
            subprocess.run(
                ["pixi", "run", "python", "-m", "rt.embed", db_name],
                cwd=rustler_dir,
                check=True,
            )


def build_pretrain_commands(cohort_names: list[str], output_dir: Path) -> list[str]:
    return [
        "pixi run torchrun --standalone --nproc_per_node=1 "
        f"scripts/synthetic_pretrain.py  # cohort={cohort_name} manifest={output_dir / 'manifest.json'}"
        for cohort_name in cohort_names
    ]


def run(
    num_dbs_per_cohort: int,
    base_seed: int,
    num_proc: int,
    stages: list[str],
    output_dir: Path,
    cache_root: Path,
) -> None:
    output_dir = output_dir.expanduser()
    cache_root = cache_root.expanduser()
    cohorts = build_edge_prior_cohorts()
    seeds_by_cohort = {
        cohort_name: [
            base_seed + cohort_idx * num_dbs_per_cohort + db_idx
            for db_idx in range(num_dbs_per_cohort)
        ]
        for cohort_idx, cohort_name in enumerate(cohorts)
    }
    db_names_by_cohort = {
        cohort_name: [db_name_for(cohort=cohort_name, seed=seed) for seed in seeds]
        for cohort_name, seeds in seeds_by_cohort.items()
    }

    if "gen" in stages:
        jobs = [
            (cohort_name, seed) for cohort_name, seeds in seeds_by_cohort.items() for seed in seeds
        ]
        worker = partial(_generate_one_from_job, cache_root=cache_root)
        with Pool(processes=num_proc) as pool:
            list(tqdm(pool.imap_unordered(worker, jobs), total=len(jobs)))

    for stage in ["pre", "embed"]:
        if stage in stages:
            for cohort_name in cohorts:
                run_preprocess_stage(db_names=db_names_by_cohort[cohort_name], stage=stage)

    if "gen" in stages:
        for cohort_name, db_names in db_names_by_cohort.items():
            measure_synthetic_topology(
                data_root=cache_root,
                db_names=db_names,
                output_dir=output_dir / f"topology_stats.{cohort_name}",
            )

    pretrain_commands = build_pretrain_commands(
        cohort_names=list(cohorts),
        output_dir=output_dir,
    )
    write_manifest(
        path=output_dir / "manifest.json",
        cohorts=cohorts,
        seeds_by_cohort=seeds_by_cohort,
        stages=stages,
        pretrain_commands=pretrain_commands,
    )


def _generate_one_from_job(job: tuple[str, int], cache_root: Path) -> str:
    cohort_name, seed = job
    return generate_one(cohort_name=cohort_name, seed=seed, cache_root=cache_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run edge-prior ablation cohorts.")
    parser.add_argument("--num_dbs_per_cohort", type=int, default=100)
    parser.add_argument("--base_seed", type=int, default=10000)
    parser.add_argument("--num_proc", type=int, default=cpu_count())
    parser.add_argument("--stages", type=str, default="gen,pre,embed")
    parser.add_argument("--output_dir", type=Path, default=Path("out/edge_prior_ablation"))
    parser.add_argument(
        "--cache_root",
        type=Path,
        default=Path("~/.cache/relbench"),
        help="RelBench cache root where rel-synthetic-* directories are written.",
    )
    args = parser.parse_args()

    run(
        num_dbs_per_cohort=args.num_dbs_per_cohort,
        base_seed=args.base_seed,
        num_proc=args.num_proc,
        stages=parse_stages(args.stages),
        output_dir=args.output_dir,
        cache_root=args.cache_root,
    )
