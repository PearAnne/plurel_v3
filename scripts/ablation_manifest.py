from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plurel.config import Choices, Config, DatabaseParams, SCMParams


@dataclass(frozen=True)
class CohortSpec:
    name: str
    config: Config
    description: str


def build_edge_prior_cohorts() -> dict[str, CohortSpec]:
    return {
        "G0_hsbm": _single_prior("G0_hsbm", "hsbm", "Baseline HSBM FK generator."),
        "G1_erdos_renyi": _single_prior(
            "G1_erdos_renyi", "erdos_renyi", "Uniform random parent assignment."
        ),
        "G2_chung_lu": _single_prior("G2_chung_lu", "chung_lu", "Power-law parent popularity."),
        "G3_dcsbm": _single_prior(
            "G3_dcsbm", "dcsbm", "Degree-corrected hierarchical block structure."
        ),
        "G4_tpa": _single_prior("G4_tpa", "tpa", "Temporal preferential attachment."),
        "G5_null_chung_lu": CohortSpec(
            name="G5_null_chung_lu",
            config=Config(
                scm_params=SCMParams(
                    topology_prior_choices=Choices(kind="set", value=["chung_lu"]),
                    edge_prior_null_rate_choices=Choices(kind="range", value=[0.05, 0.25]),
                )
            ),
            description="Chung-Lu topology with structural nullable FK edges.",
        ),
        "G6_mix": CohortSpec(
            name="G6_mix",
            config=Config(
                database_params=DatabaseParams(edge_prior_assignment_strategy="edge_level_uniform"),
                scm_params=SCMParams(
                    topology_prior_choices=Choices(kind="set", value=["chung_lu", "dcsbm", "tpa"])
                ),
            ),
            description="Per-edge uniform mixture over non-baseline topology priors.",
        ),
    }


def _single_prior(name: str, prior_kind: str, description: str) -> CohortSpec:
    return CohortSpec(
        name=name,
        config=Config(
            scm_params=SCMParams(topology_prior_choices=Choices(kind="set", value=[prior_kind]))
        ),
        description=description,
    )


def cohort_manifest_rows(
    cohorts: dict[str, CohortSpec],
    seeds_by_cohort: dict[str, list[int]],
) -> list[dict[str, Any]]:
    rows = []
    for cohort_name, spec in cohorts.items():
        config_summary = summarize_config(spec.config)
        rows.append(
            {
                "cohort": cohort_name,
                "description": spec.description,
                "seeds": seeds_by_cohort[cohort_name],
                "num_dbs": len(seeds_by_cohort[cohort_name]),
                "config": config_summary,
                "config_hash": hash_config_summary(config_summary),
            }
        )
    return rows


def summarize_config(config: Config) -> dict[str, Any]:
    return {
        "edge_prior_assignment_strategy": config.database_params.edge_prior_assignment_strategy,
        "topology_prior_choices": list(config.scm_params.topology_prior_choices.value),
        "chung_lu_gamma_choices": list(config.scm_params.chung_lu_gamma_choices.value),
        "dcsbm_theta_alpha_choices": list(config.scm_params.dcsbm_theta_alpha_choices.value),
        "dcsbm_theta_beta_choices": list(config.scm_params.dcsbm_theta_beta_choices.value),
        "dcsbm_degree_correction_strength_choices": list(
            config.scm_params.dcsbm_degree_correction_strength_choices.value
        ),
        "tpa_alpha_choices": list(config.scm_params.tpa_alpha_choices.value),
        "tpa_beta_choices": list(config.scm_params.tpa_beta_choices.value),
        "edge_prior_null_rate_choices": list(config.scm_params.edge_prior_null_rate_choices.value),
    }


def hash_config_summary(config_summary: dict[str, Any]) -> str:
    payload = json.dumps(config_summary, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def write_manifest(
    path: Path,
    cohorts: dict[str, CohortSpec],
    seeds_by_cohort: dict[str, list[int]],
    stages: list[str],
    pretrain_commands: list[str],
) -> None:
    payload = {
        "stages": stages,
        "cohorts": cohort_manifest_rows(cohorts=cohorts, seeds_by_cohort=seeds_by_cohort),
        "pretrain_commands": pretrain_commands,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
