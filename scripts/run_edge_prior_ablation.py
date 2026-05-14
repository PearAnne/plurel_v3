from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import torch
from tqdm import tqdm

from plurel.dataset import SyntheticDataset
from plurel.utils import set_random_seed

try:
    from scripts.ablation_manifest import build_edge_prior_cohorts, write_manifest
except ModuleNotFoundError:
    from ablation_manifest import build_edge_prior_cohorts, write_manifest

VALID_STAGES = {"gen", "pre", "embed", "pretrain", "eval"}
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L12-v2"
LOGGER = logging.getLogger(__name__)


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


def run_preprocess_stage(
    db_names: list[str],
    stage: str,
    cache_root: Path,
    pre_root: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rustler_dir = repo_root / "rustler"
    env = os.environ | {"PLUREL_PRE_ROOT": str(pre_root)}
    for db_name in db_names:
        if stage == "pre":
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
                env=env,
                check=True,
            )
        elif stage == "embed":
            embedding_path = pre_root / db_name / f"text_emb_{DEFAULT_EMBEDDING_MODEL}.bin"
            if embedding_path.exists():
                LOGGER.info("Skipping existing embedding for %s", db_name)
                continue
            subprocess.run(
                [sys.executable, "-m", "rt.embed", db_name],
                cwd=repo_root,
                env=env,
                check=True,
            )


def build_pretrain_commands(cohort_names: list[str], output_dir: Path) -> list[str]:
    manifest_path = output_dir / "manifest.json"
    return [
        ".pixi/envs/default/bin/python scripts/run_edge_prior_pretrain.py "
        f"--manifest {manifest_path} "
        f"--cohorts {cohort_name} "
        "--num_train_dbs 8 "
        "--num_test_dbs 2 "
        "--max_steps 401 "
        "--max_bfs_width 128 "
        "--cache_root /local/lzd/plurel_runtime/relbench "
        "--pre_root /local/lzd/plurel_runtime/pre "
        "--save_root /local/lzd/plurel_runtime/rt_smoke "
        "--dry_run"
        for cohort_name in cohort_names
    ]


def run(
    num_dbs_per_cohort: int,
    base_seed: int,
    num_proc: int,
    stages: list[str],
    output_dir: Path,
    cache_root: Path,
    pre_root: Path,
    paired_seeds: bool = True,
    selected_cohorts: list[str] | None = None,
    write_manifest_file: bool = True,
) -> None:
    output_dir = output_dir.expanduser()
    cache_root = cache_root.expanduser().resolve()
    pre_root = pre_root.expanduser().resolve()
    cohorts = build_edge_prior_cohorts()
    if selected_cohorts is not None:
        unknown = [cohort_name for cohort_name in selected_cohorts if cohort_name not in cohorts]
        if unknown:
            raise ValueError(f"Unknown cohorts: {unknown}. Valid cohorts: {sorted(cohorts)}")
        cohorts = {cohort_name: cohorts[cohort_name] for cohort_name in selected_cohorts}
    seeds_by_cohort = build_seeds_by_cohort(
        cohort_names=list(cohorts),
        num_dbs_per_cohort=num_dbs_per_cohort,
        base_seed=base_seed,
        paired_seeds=paired_seeds,
    )
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
                run_preprocess_stage(
                    db_names=db_names_by_cohort[cohort_name],
                    stage=stage,
                    cache_root=cache_root,
                    pre_root=pre_root,
                )

    if "gen" in stages:
        measure_synthetic_topology = _load_measure_synthetic_topology()
        for cohort_name, db_names in db_names_by_cohort.items():
            measure_synthetic_topology(
                data_root=cache_root,
                db_names=db_names,
                output_dir=output_dir / f"topology_stats.{cohort_name}",
            )

    if write_manifest_file:
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
            seed_pairing="paired_schema" if paired_seeds else "unpaired_legacy",
        )


def build_seeds_by_cohort(
    cohort_names: list[str],
    num_dbs_per_cohort: int,
    base_seed: int,
    paired_seeds: bool = True,
) -> dict[str, list[int]]:
    if paired_seeds:
        seeds = [base_seed + db_idx for db_idx in range(num_dbs_per_cohort)]
        return {cohort_name: list(seeds) for cohort_name in cohort_names}
    return {
        cohort_name: [
            base_seed + cohort_idx * num_dbs_per_cohort + db_idx
            for db_idx in range(num_dbs_per_cohort)
        ]
        for cohort_idx, cohort_name in enumerate(cohort_names)
    }


def _generate_one_from_job(job: tuple[str, int], cache_root: Path) -> str:
    cohort_name, seed = job
    return generate_one(cohort_name=cohort_name, seed=seed, cache_root=cache_root)


def _load_measure_synthetic_topology() -> Callable[..., None]:
    try:
        from scripts.measure_synthetic_topology import main as measure_synthetic_topology
    except ModuleNotFoundError:
        from measure_synthetic_topology import main as measure_synthetic_topology
    return measure_synthetic_topology


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run edge-prior ablation cohorts.")
    parser.add_argument("--num_dbs_per_cohort", type=int, default=100)
    parser.add_argument("--base_seed", type=int, default=10000)
    parser.add_argument("--num_proc", type=int, default=cpu_count())
    parser.add_argument("--stages", type=str, default="gen,pre,embed")
    parser.add_argument("--output_dir", type=Path, default=Path("out/edge_prior_ablation"))
    parser.add_argument(
        "--unpaired_seeds",
        action="store_true",
        help=(
            "Use the legacy seed schedule where each cohort gets a disjoint seed block. "
            "By default, cohorts share seeds so schemas are paired across priors."
        ),
    )
    parser.add_argument(
        "--cache_root",
        type=Path,
        default=Path("/local/lzd/plurel_runtime/relbench"),
        help="RelBench cache root where rel-synthetic-* directories are written.",
    )
    parser.add_argument(
        "--pre_root",
        type=Path,
        default=Path("/local/lzd/plurel_runtime/pre"),
        help="Rustler pre output root where rel-synthetic-* pre directories are written.",
    )
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=None,
        help="Optional cohort names to run. Defaults to all edge-prior cohorts.",
    )
    parser.add_argument(
        "--skip_manifest",
        action="store_true",
        help="Do not write manifest.json. Use this for preprocessing-only reruns in existing outputs.",
    )
    args = parser.parse_args()

    run(
        num_dbs_per_cohort=args.num_dbs_per_cohort,
        base_seed=args.base_seed,
        num_proc=args.num_proc,
        stages=parse_stages(args.stages),
        output_dir=args.output_dir,
        cache_root=args.cache_root,
        pre_root=args.pre_root,
        paired_seeds=not args.unpaired_seeds,
        selected_cohorts=args.cohorts,
        write_manifest_file=not args.skip_manifest,
    )
